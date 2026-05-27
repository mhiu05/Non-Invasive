import boto3
import logging
from botocore.exceptions import ClientError
from app.core.config import settings

logger = logging.getLogger(__name__)

# Boto3 client initialization
def get_s3_client():
    if not settings.storage_endpoint or not settings.storage_key or not settings.storage_secret:
        logger.warning("Storage configuration is missing. Object storage features will fail.")
        return None
        
    return boto3.client(
        "s3",
        endpoint_url=settings.storage_endpoint,
        aws_access_key_id=settings.storage_key,
        aws_secret_access_key=settings.storage_secret,
        # For MinIO or Supabase Storage, sometimes region is required, sometimes not
        # region_name="us-east-1"
    )

def upload_video(file_bytes: bytes, job_id: str, suffix: str = ".mp4") -> str:
    """Uploads video bytes to object storage and returns the key."""
    s3 = get_s3_client()
    if not s3:
        raise ValueError("S3 client is not configured")
        
    key = f"uploads/{job_id}{suffix}"
    try:
        s3.put_object(
            Bucket=settings.storage_bucket,
            Key=key,
            Body=file_bytes
        )
        logger.info(f"Successfully uploaded {key} to {settings.storage_bucket}")
        return key
    except ClientError as e:
        logger.error(f"Failed to upload {key}: {e}")
        raise e

def download_file(key: str) -> bytes:
    """Downloads a file from object storage given its key."""
    s3 = get_s3_client()
    if not s3:
        raise ValueError("S3 client is not configured")
        
    try:
        obj = s3.get_object(Bucket=settings.storage_bucket, Key=key)
        logger.info(f"Successfully downloaded {key}")
        return obj["Body"].read()
    except ClientError as e:
        logger.error(f"Failed to download {key}: {e}")
        raise e

def delete_file(key: str) -> None:
    """Deletes a file from object storage."""
    s3 = get_s3_client()
    if not s3:
        logger.warning("S3 client not configured, skipping delete")
        return
        
    try:
        s3.delete_object(Bucket=settings.storage_bucket, Key=key)
        logger.info(f"Successfully deleted {key}")
    except ClientError as e:
        logger.error(f"Failed to delete {key}: {e}")
