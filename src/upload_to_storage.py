import boto3
import os

def upload_data():
    # Connect to MinIO using the S3 protocol
    s3_client = boto3.client(
        's3',
        endpoint_url='http://minio:9000', # The name of our Docker container!
        aws_access_key_id='admin',
        aws_secret_access_key='password123'
    )

    bucket_name = 'ml-data'
    local_file = 'data/raw/car_prices_3.csv'
    s3_key = 'raw/car_prices.csv'

    # Create the bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except:
        print(f"Creating bucket: {bucket_name}")
        s3_client.create_bucket(Bucket=bucket_name)

    # Upload the file
    print(f"Uploading {local_file} to Object Storage...")
    s3_client.upload_file(local_file, bucket_name, s3_key)
    print("Upload complete!")

if __name__ == "__main__":
    upload_data()