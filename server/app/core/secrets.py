import json
from functools import lru_cache
from typing import TypedDict

import aioboto3
from botocore.exceptions import ClientError

from app.core.logging import get_logger

logger = get_logger(__name__)


class SourceCoopCredentials(TypedDict):
    """Type definition for Source Coop credentials"""

    access_key_id: str
    secret_access_key: str


class SecretsManager:
    """AWS Secrets Manager client for retrieving Source Coop credentials"""

    def __init__(self, region: str = "us-west-2") -> None:
        """Initialize Secrets Manager client."""
        self.region = region
        self.session = aioboto3.Session()

    async def get_source_coop_credentials(
        self, secret_name: str
    ) -> SourceCoopCredentials:
        """Retrieve Source Coop credentials from AWS Secrets Manager."""
        async with self.session.client(
            "secretsmanager", region_name=self.region
        ) as client:
            try:
                logger.debug(f"Retrieving secret: {secret_name}")
                response = await client.get_secret_value(SecretId=secret_name)

                secret_value = response["SecretString"]
                credentials_dict = json.loads(secret_value)

                # Validate required fields
                if "access_key_id" not in credentials_dict:
                    raise ValueError("Missing 'access_key_id' in secret")
                if "secret_access_key" not in credentials_dict:
                    raise ValueError("Missing 'secret_access_key' in secret")

                logger.info(
                    f"Successfully retrieved credentials from secret: {secret_name}"
                )
                return SourceCoopCredentials(
                    access_key_id=credentials_dict["access_key_id"],
                    secret_access_key=credentials_dict["secret_access_key"],
                )

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ResourceNotFoundException":
                    logger.error(f"Secret not found: {secret_name}")
                    raise ValueError(
                        f"Secret '{secret_name}' not found in AWS Secrets Manager"
                    ) from e
                elif error_code == "AccessDeniedException":
                    logger.error(f"Access denied to secret: {secret_name}")
                    raise ValueError(
                        f"Access denied to secret '{secret_name}'. "
                        "Check IAM permissions."
                    ) from e
                else:
                    logger.error(f"Failed to retrieve secret {secret_name}: {e}")
                    raise ValueError(
                        f"Failed to retrieve secret '{secret_name}': {error_code}"
                    ) from e

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format in secret {secret_name}: {e}")
                raise ValueError(f"Secret '{secret_name}' contains invalid JSON") from e

            except KeyError as e:
                logger.error(f"Missing required field in secret {secret_name}: {e}")
                raise ValueError(
                    f"Secret '{secret_name}' is missing required field: {e}"
                ) from e


@lru_cache(maxsize=1)
def get_secrets_manager(region: str = "us-west-2") -> SecretsManager:
    """Get cached SecretsManager instance."""
    return SecretsManager(region)
