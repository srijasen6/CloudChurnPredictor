"""
Connector for AWS S3 to extract customer data.
"""
import logging
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from io import StringIO
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET_NAME, S3_DATA_PATH

logger = logging.getLogger(__name__)

class S3Connector:
    """
    Class to connect to AWS S3 and extract data.
    """
    def __init__(self, access_key=AWS_ACCESS_KEY, secret_key=AWS_SECRET_KEY):
        """
        Initialize S3 client.
        
        Parameters:
        -----------
        access_key : str
            AWS access key ID
        secret_key : str
            AWS secret access key
        """
        self.access_key = access_key
        self.secret_key = secret_key
        
        # Initialize S3 client
        try:
            if access_key and secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key
                )
            else:
                # Use IAM role if no keys provided
                self.s3_client = boto3.client('s3')
                
            logger.info("AWS S3 client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AWS S3 client: {e}")
            raise
    
    def extract_data(self, bucket_name=S3_BUCKET_NAME, object_key=S3_DATA_PATH):
        """
        Extract data from S3 bucket.
        
        Parameters:
        -----------
        bucket_name : str
            Name of the S3 bucket
        object_key : str
            Path to the file in the bucket
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the extracted data
        """
        try:
            logger.info(f"Extracting data from S3: {bucket_name}/{object_key}")
            
            # Get the file from S3
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
            
            # Check file extension
            if object_key.endswith('.csv'):
                # Read CSV file
                data = response['Body'].read().decode('utf-8')
                df = pd.read_csv(StringIO(data))
            elif object_key.endswith('.parquet'):
                # For Parquet files, download to temp file then read
                import tempfile
                with tempfile.NamedTemporaryFile() as tmp:
                    self.s3_client.download_fileobj(bucket_name, object_key, tmp)
                    tmp.seek(0)
                    df = pd.read_parquet(tmp.name)
            else:
                raise ValueError(f"Unsupported file format: {object_key}")
            
            logger.info(f"Successfully extracted {len(df)} rows from S3")
            return df
        
        except ClientError as e:
            logger.error(f"AWS S3 error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error extracting data from S3: {e}")
            raise
    
    def list_objects(self, bucket_name=S3_BUCKET_NAME, prefix=""):
        """
        List objects in S3 bucket.
        
        Parameters:
        -----------
        bucket_name : str
            Name of the S3 bucket
        prefix : str
            Prefix to filter objects
            
        Returns:
        --------
        list
            List of objects in the bucket
        """
        try:
            logger.info(f"Listing objects in S3: {bucket_name} with prefix: {prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []
                
        except ClientError as e:
            logger.error(f"AWS S3 error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error listing objects in S3: {e}")
            raise
    
    def upload_data(self, df, bucket_name=S3_BUCKET_NAME, object_key='processed_data.csv'):
        """
        Upload data to S3 bucket.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to upload
        bucket_name : str
            Name of the S3 bucket
        object_key : str
            Path where the file will be stored in the bucket
            
        Returns:
        --------
        bool
            True if upload successful, False otherwise
        """
        try:
            logger.info(f"Uploading data to S3: {bucket_name}/{object_key}")
            
            # Convert DataFrame to CSV
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=csv_buffer.getvalue()
            )
            
            logger.info("Data uploaded successfully")
            return True
            
        except ClientError as e:
            logger.error(f"AWS S3 error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error uploading data to S3: {e}")
            return False
