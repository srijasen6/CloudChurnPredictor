"""
Tests for data extraction functionality.
"""
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from utils.aws_connector import S3Connector
from utils.gcp_connector import BigQueryConnector

class TestDataExtraction(unittest.TestCase):
    """
    Test cases for data extraction from AWS S3 and Google BigQuery.
    """
    
    @patch('boto3.client')
    def test_s3_connector_initialization(self, mock_boto_client):
        """Test S3Connector initialization."""
        # Arrange
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        # Act
        connector = S3Connector(access_key='test_key', secret_key='test_secret')
        
        # Assert
        mock_boto_client.assert_called_once_with(
            's3',
            aws_access_key_id='test_key',
            aws_secret_access_key='test_secret'
        )
        
    @patch('boto3.client')
    def test_s3_data_extraction(self, mock_boto_client):
        """Test data extraction from S3."""
        # Arrange
        mock_client = MagicMock()
        mock_response = {
            'Body': MagicMock()
        }
        mock_response['Body'].read.return_value = 'col1,col2\nval1,val2'.encode('utf-8')
        mock_client.get_object.return_value = mock_response
        mock_boto_client.return_value = mock_client
        
        # Act
        connector = S3Connector(access_key='test_key', secret_key='test_secret')
        df = connector.extract_data(bucket_name='test-bucket', object_key='test.csv')
        
        # Assert
        mock_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test.csv'
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), ['col1', 'col2'])
        self.assertEqual(df.iloc[0]['col1'], 'val1')
        
    @patch('google.cloud.bigquery.Client')
    def test_bigquery_connector_initialization(self, mock_bq_client):
        """Test BigQueryConnector initialization."""
        # Arrange
        mock_client = MagicMock()
        mock_bq_client.return_value = mock_client
        
        # Act
        connector = BigQueryConnector(project_id='test-project')
        
        # Assert
        mock_bq_client.assert_called_once_with(project='test-project')
        
    @patch('google.cloud.bigquery.Client')
    def test_bigquery_data_extraction(self, mock_bq_client):
        """Test data extraction from BigQuery."""
        # Arrange
        mock_client = MagicMock()
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame({
            'col1': ['val1'],
            'col2': ['val2']
        })
        mock_client.query.return_value = mock_query_job
        mock_bq_client.return_value = mock_client
        
        # Act
        connector = BigQueryConnector(project_id='test-project')
        df = connector.extract_data(dataset='test_dataset', table='test_table')
        
        # Assert
        mock_client.query.assert_called_once_with(
            "SELECT * FROM `test-project.test_dataset.test_table`"
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), ['col1', 'col2'])
        self.assertEqual(df.iloc[0]['col1'], 'val1')

if __name__ == '__main__':
    unittest.main()
