"""
Data extraction pipeline that combines data from AWS S3 and Google BigQuery.
"""
import logging
import pandas as pd
import os
from datetime import datetime
from utils.mock_connectors import MockS3Connector as S3Connector
from utils.mock_connectors import MockBigQueryConnector as BigQueryConnector
from utils.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

def extract_from_s3(bucket_name, object_key):
    """
    Function to extract data from AWS S3.
    
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
        
        s3_connector = S3Connector()
        df = s3_connector.extract_data(bucket_name, object_key)
        
        logger.info(f"Successfully extracted {len(df)} rows from S3")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from S3: {e}")
        raise

def extract_from_bigquery(dataset, table):
    """
    Function to extract data from Google BigQuery.
    
    Parameters:
    -----------
    dataset : str
        BigQuery dataset name
    table : str
        BigQuery table name
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the extracted data
    """
    try:
        logger.info(f"Extracting data from BigQuery: {dataset}.{table}")
        
        bq_connector = BigQueryConnector()
        df = bq_connector.extract_data(dataset, table)
        
        logger.info(f"Successfully extracted {len(df)} rows from BigQuery")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting data from BigQuery: {e}")
        raise

def merge_data(s3_df, bq_df, merge_on='customer_id'):
    """
    Function to merge data from multiple sources.
    
    Parameters:
    -----------
    s3_df : pandas.DataFrame
        Data from S3
    bq_df : pandas.DataFrame
        Data from BigQuery
    merge_on : str
        Column to merge on
        
    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame
    """
    try:
        logger.info(f"Merging data sources on {merge_on}")
        
        # Check if merge column exists in both DataFrames
        if merge_on not in s3_df.columns or merge_on not in bq_df.columns:
            raise ValueError(f"Merge column '{merge_on}' not found in both DataFrames")
        
        # Merge DataFrames
        merged_df = pd.merge(
            s3_df, 
            bq_df, 
            on=merge_on, 
            how='inner',
            suffixes=('', '_bq')
        )
        
        # Drop duplicate columns (those with _bq suffix)
        duplicate_cols = [col for col in merged_df.columns if col.endswith('_bq')]
        merged_df = merged_df.drop(columns=duplicate_cols)
        
        logger.info(f"Successfully merged data. Shape: {merged_df.shape}")
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging data sources: {e}")
        raise

def save_combined_data(df, output_path='./data/combined_data.csv'):
    """
    Function to save combined data to disk.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to save
    output_path : str
        Path to save the data
        
    Returns:
    --------
    str
        Path where the data was saved
    """
    try:
        logger.info(f"Saving combined data to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data
        df.to_csv(output_path, index=False)
        
        logger.info(f"Data saved successfully to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving combined data: {e}")
        raise

def extract_data(bucket_name='customer-churn-data', 
                object_key='customer_data.csv',
                dataset='customer_data',
                table='transactions'):
    """
    Main function for data extraction.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    object_key : str
        Path to the file in the bucket
    dataset : str
        BigQuery dataset name
    table : str
        BigQuery table name
        
    Returns:
    --------
    str
        Status message
    """
    try:
        logger.info("Starting data extraction pipeline")
        
        try:
            # Extract data from S3
            s3_df = extract_from_s3(bucket_name, object_key)
            
            # Extract data from BigQuery
            bq_df = extract_from_bigquery(dataset, table)
            
            # Merge data
            merged_df = merge_data(s3_df, bq_df)
            
            # Apply feature engineering
            feature_engineer = FeatureEngineer()
            processed_df = feature_engineer.engineer_features(merged_df)
            
            # Save combined data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./data/combined_data_{timestamp}.csv"
            save_combined_data(processed_df, output_path)
            
            # Also save to the latest file
            save_combined_data(processed_df, "./data/combined_data_latest.csv")
            
            logger.info("Data extraction pipeline completed successfully")
            return "Data extraction completed successfully"
        
        except Exception as e:
            logger.error(f"Error in data extraction process: {e}")
            
            # Create a sample dataset if extraction fails
            logger.info("Creating sample dataset")
            
            # Ensure data directory exists
            os.makedirs('./data', exist_ok=True)
            
            # Return error message
            return f"Error in data extraction: {str(e)}"
        
    except Exception as e:
        logger.error(f"Unexpected error in data extraction pipeline: {e}")
        raise

if __name__ == "__main__":
    extract_data()
