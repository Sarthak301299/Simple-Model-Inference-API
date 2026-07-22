import os
import uuid
import json
from argparse import ArgumentParser
from sagemaker.serve import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from dotenv import load_dotenv
from typing import Dict, List, Tuple, TypedDict
from sagemaker.core.helper.pipeline_variable import StrPipeVar
from sagemaker.core.resources import (
    Endpoint,
    Model,
    EndpointConfig,
    InvokeEndpointOutput,
)
from sagemaker.serve.local_resources import LocalEndpoint
from sagemaker.core.transformer import Transformer
from sagemaker.core.helper.session_helper import Session


class PredictionResponse(TypedDict):
    prediction: List[Tuple[str, float]]
    inference_time_ms: float


def create_multipart_body(
    file_bytes: bytes, filename: str, content_type: str
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode("utf-8")

    body += file_bytes
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    return body, f"multipart/form-data; boundary={boundary}"


class AWSDeployer:
    def __init__(self) -> None:
        self.AWS_ACCOUNT_ID: str = ""
        self.AWS_REGION: str = ""
        self.AWS_S3_MODEL_PATH: str = ""
        self.AWS_INSTANCE_TYPE: str = ""
        self.AWS_ENDPOINT_NAME: str = ""
        self.AWS_DEVICE_TYPE: str = ""
        self.AWS_INFERENCE_BACKEND: str = ""
        self.AWS_ROLE_ARN: str = ""
        self.AWS_ECR_NAME: str = ""
        self.model: Model | ModelBuilder | None = None
        self.endpoint: Endpoint | LocalEndpoint | Transformer | None = None
        self.model_deployed: bool = False
        self.endpoint_config_name: str = ""

        load_dotenv(".env.aws")
        self.AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", self.AWS_ACCOUNT_ID)
        self.AWS_REGION = os.getenv("AWS_REGION", self.AWS_REGION)
        self.AWS_S3_MODEL_PATH = os.getenv("AWS_S3_MODEL_PATH", self.AWS_S3_MODEL_PATH)
        self.AWS_INSTANCE_TYPE = os.getenv("AWS_INSTANCE_TYPE", self.AWS_INSTANCE_TYPE)
        self.AWS_ENDPOINT_NAME = os.getenv("AWS_ENDPOINT_NAME", self.AWS_ENDPOINT_NAME)
        self.AWS_DEVICE_TYPE = os.getenv("AWS_DEVICE_TYPE", self.AWS_DEVICE_TYPE)
        self.AWS_INFERENCE_BACKEND = os.getenv(
            "AWS_INFERENCE_BACKEND", self.AWS_INFERENCE_BACKEND
        )
        self.AWS_ROLE_ARN = os.getenv("AWS_ROLE_ARN", self.AWS_ROLE_ARN)
        self.AWS_ECR_NAME = os.getenv("AWS_ECR_NAME", self.AWS_ECR_NAME)

        if len(self.AWS_ACCOUNT_ID) != 12:
            raise ValueError("AWS_ACCOUNT_ID must be a 12-digit AWS account ID")
        if self.AWS_REGION == "":
            raise ValueError("AWS_REGION must be a valid AWS region")
        if self.AWS_S3_MODEL_PATH == "":
            raise ValueError("AWS_S3_MODEL_PATH is not provided")
        if self.AWS_INSTANCE_TYPE == "":
            raise ValueError("AWS_INSTANCE_TYPE is not provided")
        if self.AWS_ENDPOINT_NAME == "":
            raise ValueError("AWS_ENDPOINT_NAME is not provided")
        if self.AWS_DEVICE_TYPE not in ("cpu", "gpu"):
            raise ValueError("AWS_DEVICE_TYPE must be cpu or gpu")
        if self.AWS_INFERENCE_BACKEND not in ("torch", "onnx", "tensorrt"):
            raise ValueError("AWS_INFERENCE_BACKEND must be torch, onnx, or tensorrt")
        if self.AWS_ROLE_ARN == "":
            raise ValueError("AWS_ROLE_ARN is not provided")
        if self.AWS_ECR_NAME == "":
            raise ValueError("AWS_ECR_NAME is not provided")

    def deploy_model_to_sagemaker(self) -> None:
        # Define your ECR Image URI
        image_uri = f"{self.AWS_ACCOUNT_ID}.dkr.ecr.{self.AWS_REGION}.amazonaws.com/{self.AWS_ECR_NAME}:aws-{self.AWS_DEVICE_TYPE}"

        env_vars: Dict[str, StrPipeVar] = {
            "API_KEY": "",
            "INFERENCE_BACKEND": f"{self.AWS_INFERENCE_BACKEND}",
            "INFERENCE_DEVICE": "cuda" if self.AWS_DEVICE_TYPE == "gpu" else "cpu",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8080",
            "HF_HOME": "/home/modelapi/.cache/huggingface",
            "MODEL_PATH": (
                "/opt/ml/model" if self.AWS_INFERENCE_BACKEND != "torch" else ""
            ),
        }

        # 1. Define the Model object
        try:
            self.session = Session()
            model_builder = ModelBuilder(
                image_uri=image_uri,
                role_arn=self.AWS_ROLE_ARN,
                env_vars=env_vars,
                s3_model_data_url=f"{self.AWS_S3_MODEL_PATH}/model.tar.gz",
                sagemaker_session=self.session,
            )
            self.model = model_builder.build(mode=Mode.SAGEMAKER_ENDPOINT)
            if not isinstance(self.model, Model):
                raise TypeError("ModelBuilder build returned incorrect datatype")
        except Exception as e:
            print(f"Error initializing ModelBuilder: {e}")
            raise

        # 2. Deploy to a Real-Time Endpoint
        try:
            self.endpoint = model_builder.deploy(
                endpoint_name=self.AWS_ENDPOINT_NAME,
                instance_type=self.AWS_INSTANCE_TYPE,
                container_timeout_in_seconds=90,
            )
            if not isinstance(self.endpoint, Endpoint):
                raise TypeError("ModelBuilder deploy returned incorrect datatype")
            if self.endpoint.endpoint_name != self.AWS_ENDPOINT_NAME:
                raise ValueError("Endpoint name does not match input name")
            epdesc = self.session.sagemaker_client.describe_endpoint(
                EndpointName=self.endpoint.endpoint_name
            )
            self.endpoint_config_name = epdesc["EndpointConfigName"]
        except Exception:
            print("Error deploying model")
            try:
                if isinstance(self.endpoint, Endpoint):
                    self.endpoint.delete()
            except Exception as cleanup_error:
                print(f"Endpoint cleanup failed: {cleanup_error}")
            try:
                if self.endpoint_config_name:
                    endpoint_config = EndpointConfig.get(
                        endpoint_config_name=self.endpoint_config_name
                    )
                    if endpoint_config is not None:
                        endpoint_config.delete()
                else:
                    endpoint_config = EndpointConfig.get(
                        endpoint_config_name=self.AWS_ENDPOINT_NAME
                    )
                    if endpoint_config is not None:
                        endpoint_config.delete()
            except Exception as cleanup_error:
                print(f"Endpoint config cleanup failed: {cleanup_error}")
            try:
                if isinstance(self.model, Model):
                    self.model.delete()
            except Exception as cleanup_error:
                print(f"Model cleanup failed: {cleanup_error}")
            raise
        self.model_deployed = True
        print(f"Endpoint deployed successfully: {self.endpoint.endpoint_name}")

    def delete_endpoint(self) -> None:
        if self.model_deployed:
            errors = []
            endpoint_config: EndpointConfig | None = None
            try:
                endpoint_config: EndpointConfig | None = EndpointConfig.get(
                    endpoint_config_name=self.endpoint_config_name
                )
            except Exception as e:
                errors.append(("Error Getting endpoint_config", e))
            try:
                if isinstance(self.endpoint, Endpoint):
                    self.endpoint.delete()
            except Exception as e:
                errors.append(("Error in endpoint deletion", e))
            try:
                if endpoint_config is not None and isinstance(
                    endpoint_config, EndpointConfig
                ):
                    endpoint_config.delete()
            except Exception as e:
                errors.append(("Error in endpoint_config deletion", e))
            try:
                if isinstance(self.model, Model):
                    self.model.delete()
            except Exception as e:
                errors.append(("Error in model deletion", e))
            if errors:
                error_summary = "\n".join(
                    [f"{loc}: {type(err).__name__} - {err}" for loc, err in errors]
                )
                raise RuntimeError(
                    f"Execution completed with {len(errors)} error(s):\n{error_summary}"
                )
        self.model_deployed = False

    def make_prediction(
        self, input_data: bytes, content_type: str
    ) -> PredictionResponse:
        if not self.model_deployed:
            raise RuntimeError("Model is not deployed on AWS.")

        if not isinstance(self.endpoint, Endpoint):
            raise TypeError("Endpoint must be an instance of Endpoint")

        if not content_type.startswith("multipart/form-data;"):
            raise ValueError("Content-Type must be multipart/form-data with a boundary")

        try:
            response: InvokeEndpointOutput = self.endpoint.invoke(
                body=input_data, content_type=content_type, accept="application/json"
            )
        except Exception as e:
            print(f"Exception during invoke {e}")
            raise
        if not isinstance(response, InvokeEndpointOutput):
            raise TypeError("Endpoint invocation returned an unexpected response type")
        body = response.body
        if hasattr(body, "read"):
            body = body.read()
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        output: PredictionResponse = json.loads(body)
        if not isinstance(output, dict):
            raise TypeError("Inference response must be a JSON object")
        return output


def main():
    parser = ArgumentParser(
        description="Deploy a model to AWS SageMaker and make predictions."
    )
    parser.add_argument(
        "--file", type=str, help="Image File to be passed as input.", required=True
    )
    args = parser.parse_args()

    awsdeployer: AWSDeployer | None = None
    try:
        awsdeployer = AWSDeployer()
        awsdeployer.deploy_model_to_sagemaker()
        with open(args.file, "rb") as file:
            file_bytes = file.read()
            multipart_body, content_type = create_multipart_body(
                file_bytes=file_bytes,
                filename=os.path.basename(args.file),
                content_type="image/jpeg",
            )
            output: PredictionResponse = awsdeployer.make_prediction(
                input_data=multipart_body, content_type=content_type
            )
            print(f"Received response {output}")
    finally:
        if awsdeployer is not None:
            try:
                awsdeployer.delete_endpoint()
            except Exception as cleanup_error:
                print(f"Delete Endpoint failed {cleanup_error}")


if __name__ == "__main__":
    main()  # pragma: no cover
