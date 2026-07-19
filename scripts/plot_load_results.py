"""Parses results/<platform>/<backend>/run_*users_*.csv (Locust output) and
generates comparison graphs across platform/backend combinations and
concurrency levels.

Usage:
    uv run python scripts/plot_load_results.py
    uv run python scripts/plot_load_results.py --results-dir results --output-dir results/plots

Expects the directory layout produced by the load-testing runs:
    results/<platform>/<backend>/run_<N>users_stats.csv
    results/<platform>/<backend>/run_<N>users_stats_history.csv
    results/<platform>/<backend>/run_<N>users_failures.csv
    results/<platform>/<backend>/run_<N>users_exceptions.csv

Important note on "throughput": the locustfile's predict_invalid task
(weight 1, vs. weight 5 for predict) deliberately sends bad images expecting
a 400/429, and the locustfile also treats a real 429 (queue-full backpressure)
as resp.success(). Both cases inflate the raw /predict "Requests/s" number
without representing a served prediction. This script computes throughput
from the INFER custom metric instead (fired only on an actual successful
prediction), which is the number that should be used for cross-backend
throughput comparisons — the raw POST /predict rate is included on its own
chart for transparency, but should not be read as "predictions per second".
"""

import argparse
import re
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import numpy

# Fixed across all runs in this project's load-testing setup (see
# tests/load/locustfile.py invocation) — used to convert the INFER row's
# request count into a rate. If you change --run-time, update this or pass
# --run-time-seconds.
DEFAULT_RUN_TIME_SECONDS = 116


def load_all_stats(results_dir: Path) -> pd.DataFrame:
    """Parses every run_<N>users_stats.csv under results_dir into one long
    DataFrame with columns: platform, backend, users, type, name, count,
    fail_count, median, avg, p95, p99, rps, fail_rps.
    """
    rows = []
    for stats_file in sorted(results_dir.glob("*/*/run_*users_stats.csv")):
        platform, backend = stats_file.parts[-3], stats_file.parts[-2]
        match = re.search(r"run_(\d+)users", stats_file.name)
        if not match:
            continue
        users = int(match.group(1))

        df = pd.read_csv(stats_file, engine="python")
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.iterrows():
            rows.append(
                {
                    "platform": platform,
                    "backend": backend,
                    "combo": f"{platform}/{backend}",
                    "users": users,
                    "type": r["Type"],
                    "name": r["Name"],
                    "count": r["Request Count"],
                    "fail_count": r["Failure Count"],
                    "median": r["Median Response Time"],
                    "avg": r["Average Response Time"],
                    "p95": r["95%"],
                    "p99": r["99%"],
                    "rps": r["Requests/s"],
                    "fail_rps": r["Failures/s"],
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"No run_*users_stats.csv files found under {results_dir}. "
            "Expected results/<platform>/<backend>/run_<N>users_stats.csv"
        )
    return pd.DataFrame(rows)


def compute_real_throughput(df: pd.DataFrame, run_time_seconds: float) -> pd.DataFrame:
    """Real, achieved throughput in successful predictions/sec, derived from
    the INFER row's request count — NOT the raw POST /predict Requests/s
    column, which includes intentionally-invalid requests and backpressure
    429s. See module docstring."""
    infer = df[df["name"] == "inference_time_ms"].copy()
    infer["real_rps"] = infer["count"] / run_time_seconds
    return infer.pivot_table(index="users", columns="combo", values="real_rps")


def compute_rejection_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of POST /predict requests that never produced an INFER
    event — i.e., never reached a successful prediction. At low/medium
    concurrency this should sit close to 1/6 (~16.7%), matching the
    locustfile's deliberate predict_invalid task weight (1 invalid : 5 valid).
    A ratio meaningfully above that baseline indicates real backpressure
    (429s) or failures on top of the expected invalid-image traffic."""
    post = df[df["name"] == "/predict"].pivot_table(
        index="users", columns="combo", values="count"
    )
    infer_count = df[df["name"] == "inference_time_ms"].pivot_table(
        index="users", columns="combo", values="count"
    )
    return cast(pd.DataFrame, (1 - infer_count / post) * 100)


def plot_throughput(real_throughput: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for combo in real_throughput.columns:
        ax.plot(real_throughput.index, real_throughput[combo], marker="o", label=combo)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("Successful predictions / sec")
    ax.set_title(
        "Real achieved throughput vs. concurrency\n(derived from INFER events, not raw request count)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_raw_post_rps(df: pd.DataFrame, output_dir: Path) -> None:
    """Included for transparency/comparison against plot_throughput — this is
    what you'd get from Locust's own Aggregated row, and demonstrates why it
    should not be used alone for cross-backend throughput comparisons."""
    post_rps = df[df["name"] == "Aggregated"].pivot_table(
        index="users", columns="combo", values="rps"
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for combo in post_rps.columns:
        ax.plot(post_rps.index, post_rps[combo], marker="o", label=combo)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("Requests / sec (raw, includes invalid + 429)")
    ax.set_title(
        "Raw POST /predict throughput vs. concurrency\n"
        "(includes intentional bad-image traffic and backpressure 429s — see throughput_vs_concurrency.png instead)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "raw_post_rps_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_latency_percentile(
    df: pd.DataFrame, output_dir: Path, percentile: str
) -> None:
    """percentile: '95%' or '99%' — plots INFER (pure model compute) latency."""
    col = "p95" if percentile == "95%" else "p99"
    data = df[df["name"] == "inference_time_ms"].pivot_table(
        index="users", columns="combo", values=col
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for combo in data.columns:
        ax.plot(data.index, data[combo], marker="o", label=combo)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel(f"Inference time {percentile} (ms, log scale)")
    ax.set_title(f"Model inference latency ({percentile}) vs. concurrency")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"inference_latency_{col}_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_queueing_overhead(df: pd.DataFrame, output_dir: Path) -> None:
    """WALL p95 (client-observed round trip) minus INFER p95 (pure model
    compute) isolates queueing/batching-wait overhead from raw compute time."""
    wall = df[df["name"] == "predict_wall_time_ms"].pivot_table(
        index="users", columns="combo", values="p95"
    )
    infer = df[df["name"] == "inference_time_ms"].pivot_table(
        index="users", columns="combo", values="p95"
    )
    overhead = wall - infer

    fig, ax = plt.subplots(figsize=(9, 6))
    for combo in overhead.columns:
        ax.plot(overhead.index, overhead[combo], marker="o", label=combo)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("WALL p95 - INFER p95 (ms)")
    ax.set_title(
        "Queueing / batching-wait overhead vs. concurrency\n(client-observed latency minus pure model compute time)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "queueing_overhead_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_rejection_ratio(rejection_ratio: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for combo in rejection_ratio.columns:
        ax.plot(rejection_ratio.index, rejection_ratio[combo], marker="o", label=combo)
    ax.axhline(
        100 / 6,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Expected baseline (predict_invalid task weight, ~16.7%)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("% of /predict requests without a successful prediction")
    ax.set_title(
        "Rejection ratio vs. concurrency\n(above the dashed line = real backpressure/failures, not just intentional bad-image traffic)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rejection_ratio_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_peak_throughput_bar(real_throughput: pd.DataFrame, output_dir: Path) -> None:
    peak = real_throughput.max().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(peak.index, numpy.asarray(peak.values), color="steelblue")
    ax.set_ylabel("Peak successful predictions / sec")
    ax.set_title("Peak real throughput by platform/backend")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(peak.values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "peak_throughput_by_combo.png", dpi=150)
    plt.close(fig)


def write_summary_csv(
    df: pd.DataFrame,
    real_throughput: pd.DataFrame,
    rejection_ratio: pd.DataFrame,
    output_dir: Path,
) -> None:
    """One row per (combo, users) with the key numbers side by side — useful
    for a quick scan or pasting into a report table without opening the plots."""
    infer = df[df["name"] == "inference_time_ms"].set_index(["users", "combo"])
    summary = pd.DataFrame(
        {
            "real_throughput_rps": real_throughput.stack(),
            "rejection_ratio_pct": rejection_ratio.stack(),
        }
    )
    summary["infer_p95_ms"] = infer["p95"]
    summary["infer_p99_ms"] = infer["p99"]
    summary = summary.reset_index().rename(
        columns={"level_0": "users", "level_1": "combo"}
    )
    summary.to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--output-dir", default="results/plots", type=Path)
    parser.add_argument(
        "--run-time-seconds", type=float, default=DEFAULT_RUN_TIME_SECONDS
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_stats(args.results_dir)
    real_throughput = compute_real_throughput(df, args.run_time_seconds)
    rejection_ratio = compute_rejection_ratio(df)

    plot_throughput(real_throughput, args.output_dir)
    plot_raw_post_rps(df, args.output_dir)
    plot_latency_percentile(df, args.output_dir, "95%")
    plot_latency_percentile(df, args.output_dir, "99%")
    plot_queueing_overhead(df, args.output_dir)
    plot_rejection_ratio(rejection_ratio, args.output_dir)
    plot_peak_throughput_bar(real_throughput, args.output_dir)
    write_summary_csv(df, real_throughput, rejection_ratio, args.output_dir)

    print(f"Wrote plots and summary.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
