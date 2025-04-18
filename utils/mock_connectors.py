"""
Mock connectors for AWS S3 and Google BigQuery to use for development and testing.
"""
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MockS3Connector:
    """
    Mock class to simulate AWS S3 connector.
    """
    def __init__(self, access_key=None, secret_key=None):
        """
        Initialize mock S3 client.
        
        Parameters:
        -----------
        access_key : str
            AWS access key ID (not used in mock)
        secret_key : str
            AWS secret access key (not used in mock)
        """
        logger.info("Initialized mock AWS S3 connector")
    
    def extract_data(self, bucket_name=None, object_key=None):
        """
        Extract mock data instead of from S3 bucket.
        
        Parameters:
        -----------
        bucket_name : str
            Name of the S3 bucket (not used in mock)
        object_key : str
            Path to the file in the bucket (not used in mock)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing mock data
        """
        logger.info(f"Mock extracting data from S3: {bucket_name}/{object_key}")
        
        # Create a mock dataset for customer data
        np.random.seed(42)
        n_samples = 100
        
        # Generate customer IDs
        customer_ids = [f"CUST{i:06d}" for i in range(1, n_samples + 1)]
        
        # Generate demographics
        genders = np.random.choice(['Male', 'Female'], size=n_samples)
        ages = np.random.randint(18, 75, size=n_samples)
        tenures = np.random.randint(1, 72, size=n_samples)
        monthly_charges = np.random.uniform(30, 120, size=n_samples).round(2)
        total_charges = (monthly_charges * tenures * np.random.uniform(0.9, 1.1, size=n_samples)).round(2)
        
        # Generate contract information
        contract_types = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            size=n_samples, 
            p=[0.6, 0.3, 0.1]
        )
        payment_methods = np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            size=n_samples,
            p=[0.4, 0.2, 0.3, 0.1]
        )
        internet_services = np.random.choice(
            ['DSL', 'Fiber optic', 'No'],
            size=n_samples,
            p=[0.3, 0.6, 0.1]
        )
        
        # Additional features
        dependents = np.random.randint(0, 5, size=n_samples)
        referrals = np.random.randint(0, 8, size=n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'gender': genders,
            'age': ages,
            'tenure': tenures,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_type': contract_types,
            'payment_method': payment_methods,
            'internet_service': internet_services,
            'number_of_dependents': dependents,
            'number_of_referrals': referrals
        })
        
        logger.info(f"Successfully extracted {len(df)} rows from mock S3")
        return df
    
    def list_objects(self, bucket_name=None, prefix=""):
        """
        List mock objects instead of from S3 bucket.
        
        Parameters:
        -----------
        bucket_name : str
            Name of the S3 bucket (not used in mock)
        prefix : str
            Prefix to filter objects (not used in mock)
            
        Returns:
        --------
        list
            List of mock objects
        """
        logger.info(f"Mock listing objects in S3: {bucket_name} with prefix: {prefix}")
        return ['customer_data.csv', 'transactions.csv', 'demographics.csv']
    
    def upload_data(self, df, bucket_name=None, object_key='processed_data.csv'):
        """
        Mock upload data to S3 bucket.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to upload
        bucket_name : str
            Name of the S3 bucket (not used in mock)
        object_key : str
            Path where the file will be stored in the bucket (not used in mock)
            
        Returns:
        --------
        bool
            True if mock upload successful
        """
        logger.info(f"Mock uploading data to S3: {bucket_name}/{object_key}")
        return True


class MockBigQueryConnector:
    """
    Mock class to simulate Google BigQuery connector.
    """
    def __init__(self, project_id=None, credentials_path=None):
        """
        Initialize mock BigQuery client.
        
        Parameters:
        -----------
        project_id : str
            Google Cloud project ID (not used in mock)
        credentials_path : str
            Path to the service account key JSON file (not used in mock)
        """
        logger.info("Initialized mock Google BigQuery connector")
    
    def extract_data(self, dataset=None, table=None, limit=None):
        """
        Extract mock data instead of from BigQuery table.
        
        Parameters:
        -----------
        dataset : str
            BigQuery dataset name (not used in mock)
        table : str
            BigQuery table name (not used in mock)
        limit : int, optional
            Maximum number of rows to extract (not used in mock)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing mock data
        """
        logger.info(f"Mock extracting data from BigQuery: {dataset}.{table}")
        
        # Create a mock dataset for transaction data
        np.random.seed(43)
        n_samples = 100
        
        # Generate customer IDs (same as in S3 mock data)
        customer_ids = [f"CUST{i:06d}" for i in range(1, n_samples + 1)]
        
        # Generate transaction dates
        base_date = datetime(2023, 1, 1)
        transaction_dates = [
            (base_date + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
            for _ in range(n_samples)
        ]
        
        # Generate transaction amounts
        transaction_amounts = np.random.uniform(10, 200, size=n_samples).round(2)
        
        # Generate transaction types
        transaction_types = np.random.choice(
            ['Purchase', 'Refund', 'Subscription', 'Upgrade'],
            size=n_samples,
            p=[0.7, 0.1, 0.15, 0.05]
        )
        
        # Generate channels
        channels = np.random.choice(
            ['Web', 'Mobile', 'Phone', 'In-Store'],
            size=n_samples,
            p=[0.5, 0.3, 0.1, 0.1]
        )
        
        # Create churn indicator
        churn = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'transaction_date': transaction_dates,
            'transaction_amount': transaction_amounts,
            'transaction_type': transaction_types,
            'channel': channels,
            'churn': churn
        })
        
        logger.info(f"Successfully extracted {len(df)} rows from mock BigQuery")
        return df
    
    def extract_data_with_query(self, query):
        """
        Extract mock data using a custom SQL query.
        
        Parameters:
        -----------
        query : str
            SQL query to execute (not used in mock)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing mock query results
        """
        logger.info(f"Mock executing custom query on BigQuery: {query[:100]}...")
        
        # Just return the same mock data as extract_data
        return self.extract_data()
    
    def list_tables(self, dataset=None):
        """
        List mock tables instead of from BigQuery dataset.
        
        Parameters:
        -----------
        dataset : str
            BigQuery dataset name (not used in mock)
            
        Returns:
        --------
        list
            List of mock table names
        """
        logger.info(f"Mock listing tables in dataset: {dataset}")
        return ['transactions', 'customers', 'products', 'events']
    
    def upload_data(self, df, dataset=None, table="processed_data"):
        """
        Mock upload data to BigQuery table.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to upload
        dataset : str
            BigQuery dataset name (not used in mock)
        table : str
            BigQuery table name (not used in mock)
            
        Returns:
        --------
        bool
            True if mock upload successful
        """
        logger.info(f"Mock uploading data to BigQuery: {dataset}.{table}")
        return True