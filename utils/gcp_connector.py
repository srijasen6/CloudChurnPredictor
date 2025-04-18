"""
Connector for Google BigQuery to extract customer data.
"""
import logging
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import os
from config import GCP_PROJECT_ID, GCP_CREDENTIALS_PATH, BQ_DATASET, BQ_TABLE

logger = logging.getLogger(__name__)

class BigQueryConnector:
    """
    Class to connect to Google BigQuery and extract data.
    """
    def __init__(self, project_id=GCP_PROJECT_ID, credentials_path=GCP_CREDENTIALS_PATH):
        """
        Initialize BigQuery client.
        
        Parameters:
        -----------
        project_id : str
            Google Cloud project ID
        credentials_path : str
            Path to the service account key JSON file
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        
        # Initialize BigQuery client
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                self.bq_client = bigquery.Client(
                    project=project_id,
                    credentials=credentials
                )
            else:
                # Use application default credentials
                self.bq_client = bigquery.Client(project=project_id)
                
            logger.info("Google BigQuery client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Google BigQuery client: {e}")
            raise
    
    def extract_data(self, dataset=BQ_DATASET, table=BQ_TABLE, limit=None):
        """
        Extract data from BigQuery table.
        
        Parameters:
        -----------
        dataset : str
            BigQuery dataset name
        table : str
            BigQuery table name
        limit : int, optional
            Maximum number of rows to extract
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the extracted data
        """
        try:
            logger.info(f"Extracting data from BigQuery: {dataset}.{table}")
            
            # Build query
            query = f"SELECT * FROM `{self.project_id}.{dataset}.{table}`"
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            df = self.bq_client.query(query).to_dataframe()
            
            logger.info(f"Successfully extracted {len(df)} rows from BigQuery")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting data from BigQuery: {e}")
            raise
    
    def extract_data_with_query(self, query):
        """
        Extract data using a custom SQL query.
        
        Parameters:
        -----------
        query : str
            SQL query to execute
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the query results
        """
        try:
            logger.info(f"Executing custom query on BigQuery: {query[:100]}...")
            
            # Execute query
            df = self.bq_client.query(query).to_dataframe()
            
            logger.info(f"Successfully extracted {len(df)} rows from BigQuery")
            return df
            
        except Exception as e:
            logger.error(f"Error executing custom query on BigQuery: {e}")
            raise
    
    def list_tables(self, dataset=BQ_DATASET):
        """
        List tables in a BigQuery dataset.
        
        Parameters:
        -----------
        dataset : str
            BigQuery dataset name
            
        Returns:
        --------
        list
            List of table names in the dataset
        """
        try:
            logger.info(f"Listing tables in dataset: {dataset}")
            
            tables = list(self.bq_client.list_tables(f"{self.project_id}.{dataset}"))
            table_names = [table.table_id for table in tables]
            
            return table_names
            
        except Exception as e:
            logger.error(f"Error listing tables in dataset: {e}")
            raise
    
    def upload_data(self, df, dataset=BQ_DATASET, table="processed_data"):
        """
        Upload data to BigQuery table.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to upload
        dataset : str
            BigQuery dataset name
        table : str
            BigQuery table name
            
        Returns:
        --------
        bool
            True if upload successful, False otherwise
        """
        try:
            logger.info(f"Uploading data to BigQuery: {dataset}.{table}")
            
            # Define table reference
            table_ref = f"{self.project_id}.{dataset}.{table}"
            
            # Upload DataFrame to BigQuery
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE"
            )
            
            job = self.bq_client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )
            
            # Wait for job to complete
            job.result()
            
            logger.info(f"Data uploaded successfully to {table_ref}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading data to BigQuery: {e}")
            return False
