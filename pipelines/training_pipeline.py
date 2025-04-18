"""
Model training pipeline for customer churn prediction.
"""
import logging
import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from utils.data_processor import DataProcessor
from model.model_trainer import train_xgboost_model
from config import MODEL_OUTPUT_PATH, MODEL_METRICS_PATH, FEATURE_IMPORTANCE_PATH

logger = logging.getLogger(__name__)

def load_data(data_path='./data/combined_data_latest.csv'):
    """
    Function to load the combined data for model training.
    
    Parameters:
    -----------
    data_path : str
        Path to the combined data file
        
    Returns:
    --------
    pandas.DataFrame
        Combined data for model training
    """
    try:
        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        logger.info(f"Successfully loaded {len(df)} rows from {data_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def process_data(df):
    """
    Function to process the data for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to process
        
    Returns:
    --------
    dict
        Dictionary containing processed data
    """
    try:
        logger.info("Processing data for model training")
        
        # Create data processor
        data_processor = DataProcessor()
        
        # Process data
        processed_data = data_processor.prepare_data(df, train=True)
        
        # Save preprocessor for later use
        data_processor.save_preprocessor()
        
        logger.info("Data processing complete")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def get_default_params():
    """
    Function to get default hyperparameters.
    
    Returns:
    --------
    dict
        Dictionary containing default hyperparameters
    """
    return {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'min_child_weight': 1
    }

def train_model_with_params(processed_data, params=None):
    """
    Function to train the XGBoost model.
    
    Parameters:
    -----------
    processed_data : dict
        Dictionary containing processed data
    params : dict, optional
        Dictionary containing hyperparameters
        
    Returns:
    --------
    xgboost.XGBClassifier
        Trained model
    """
    try:
        logger.info("Training XGBoost model")
        
        # Extract data
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        # Use default params if none provided
        if params is None:
            params = get_default_params()
            logger.info("Using default hyperparameters")
        
        # Train model
        model = train_xgboost_model(X_train, y_train, params)
        
        logger.info("Model training complete")
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, processed_data):
    """
    Function to evaluate the trained model.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained model
    processed_data : dict
        Dictionary containing processed data
        
    Returns:
    --------
    dict
        Dictionary containing model metrics
    """
    try:
        logger.info("Evaluating model performance")
        
        # Extract data
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'auc_roc': float(roc_auc_score(y_test, y_prob[:, 1])),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Model metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def save_model(model, metrics, processed_data, params):
    """
    Function to save the trained model and metrics.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained model
    metrics : dict
        Dictionary containing model metrics
    processed_data : dict
        Dictionary containing processed data
    params : dict
        Dictionary containing hyperparameters
        
    Returns:
    --------
    dict
        Dictionary containing paths to saved files
    """
    try:
        logger.info("Saving model and metrics")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        
        # Save model
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        with open(MODEL_METRICS_PATH, 'w') as f:
            json.dump(metrics, f)
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = processed_data.get('feature_names', [])
            
            # If feature names aren't available, use generic names
            if not feature_names or len(feature_names) != len(model.feature_importances_):
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            
            # Create feature importance dictionary
            importance_values = model.feature_importances_
            feature_importance = {
                name: float(importance) 
                for name, importance in zip(feature_names, importance_values)
            }
            
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # Save feature importance
            with open(FEATURE_IMPORTANCE_PATH, 'w') as f:
                json.dump(feature_importance, f)
        
        logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
        logger.info(f"Metrics saved to {MODEL_METRICS_PATH}")
        
        return {
            'model_path': MODEL_OUTPUT_PATH,
            'metrics_path': MODEL_METRICS_PATH,
            'feature_importance_path': FEATURE_IMPORTANCE_PATH if feature_importance else None
        }
        
    except Exception as e:
        logger.error(f"Error saving model and metrics: {e}")
        raise

def train_model(data_path=None):
    """
    Main function for model training.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file
        
    Returns:
    --------
    str
        Status message
    """
    try:
        logger.info("Starting model training")
        
        # Set default data path if not provided
        if data_path is None:
            data_path = './data/combined_data_latest.csv'
            
            # Check if file exists, otherwise use sample data
            if not os.path.exists(data_path):
                # Create sample data directory if it doesn't exist
                os.makedirs('./data', exist_ok=True)
                
                # Return a meaningful error
                return "Error: Training data not found. Please run data extraction first."
        
        # Load data
        df = load_data(data_path)
        
        # Process data
        processed_data = process_data(df)
        
        # Use default hyperparameters
        params = get_default_params()
        
        # Train model
        model = train_model_with_params(processed_data, params)
        
        # Evaluate model
        metrics = evaluate_model(model, processed_data)
        
        # Save model and metrics
        save_model(model, metrics, processed_data, params)
        
        logger.info("Model training completed successfully")
        
        return "Model training completed successfully"
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

if __name__ == "__main__":
    train_model()
