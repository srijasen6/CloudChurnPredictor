"""
Configuration settings for the churn prediction pipeline.
"""
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# AWS Configuration
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'customer-churn-data')
S3_DATA_PATH = os.getenv('S3_DATA_PATH', 'customer_data.csv')

# GCP Configuration
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
GCP_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BQ_DATASET = os.getenv('BQ_DATASET', 'customer_data')
BQ_TABLE = os.getenv('BQ_TABLE', 'transactions')

# Model Configuration
MODEL_OUTPUT_PATH = os.getenv('MODEL_OUTPUT_PATH', './model/model.pkl')
FEATURE_IMPORTANCE_PATH = os.getenv('FEATURE_IMPORTANCE_PATH', './model/feature_importance.json')
MODEL_METRICS_PATH = os.getenv('MODEL_METRICS_PATH', './model/metrics.json')

# Prefect Configuration
PREFECT_API_KEY = os.getenv('PREFECT_API_KEY')

# Application Configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Features for the model
TARGET_COLUMN = 'churn'
CATEGORICAL_FEATURES = [
    'gender', 
    'contract_type', 
    'payment_method', 
    'internet_service'
]
NUMERICAL_FEATURES = [
    'tenure', 
    'monthly_charges', 
    'total_charges', 
    'age', 
    'number_of_dependents', 
    'number_of_referrals'
]

# Hyperparameter tuning
N_TRIALS = 100  # Number of Optuna trials
TIMEOUT = 3600  # Seconds before Optuna times out (1 hour)
CROSS_VALIDATION_FOLDS = 5
