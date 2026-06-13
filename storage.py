import boto3
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

def upload_to_s3(file_bytes: bytes, original_filename: str) -> dict:
    bucket_name = os.getenv("S3_BUCKET_NAME")
    s3_client = get_s3_client()
    
    doc_id = str(uuid.uuid4())
    s3_key = f"documents/{doc_id}/{original_filename}"

    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=file_bytes,
        ContentType="application/pdf",
    )

    file_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

    return {
        "doc_id": doc_id,
        "s3_key": s3_key,
        "file_url": file_url,
    }

def download_from_s3(s3_key: str, local_path: str) -> str:
    s3_client = get_s3_client()
    bucket_name = os.getenv("S3_BUCKET_NAME")
    s3_client.download_file(bucket_name, s3_key, local_path)
    return local_path

def delete_from_s3(s3_key: str) -> bool:
    s3_client = get_s3_client()
    bucket_name = os.getenv("S3_BUCKET_NAME")
    s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
    return True