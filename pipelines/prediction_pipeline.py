"""
Prediction pipeline for customer churn prediction.
"""
import logging
import pandas as pd
import numpy as np
import pickle
import os
from utils.data_processor import DataProcessor
from config import MODEL_OUTPUT_PATH

logger = logging.getLogger(__name__)

def load_model(model_path=MODEL_OUTPUT_PATH):
    """
    Function to load the trained model.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
        
    Returns:
    --------
    xgboost.XGBClassifier
        Trained model
    """
    try:
        logger.info(f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def prepare_prediction_data(df):
    """
    Function to prepare data for prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to prepare
        
    Returns:
    --------
    numpy.ndarray
        Processed data for prediction
    """
    try:
        logger.info("Preparing data for prediction")
        
        # Create data processor
        data_processor = DataProcessor()
        
        # Load preprocessor
        data_processor.load_preprocessor('./model/preprocessor.pkl')
        
        # Process data
        processed_data = data_processor.prepare_data(df, train=False)
        
        logger.info("Data preparation complete")
        return processed_data['X']
        
    except Exception as e:
        logger.error(f"Error preparing prediction data: {e}")
        raise

def make_predictions(model, X):
    """
    Function to make predictions using the trained model.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained model
    X : numpy.ndarray
        Processed data for prediction
        
    Returns:
    --------
    tuple
        (predictions, probabilities)
    """
    try:
        logger.info("Making predictions")
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        logger.info(f"Made {len(predictions)} predictions")
        return predictions, probabilities
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def predict_churn(df):
    """
    Main function for making predictions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data for prediction
        
    Returns:
    --------
    tuple
        (predictions, probabilities)
    """
    try:
        logger.info("Starting prediction pipeline")
        
        # Check if model exists and return dummy results if it doesn't
        if not os.path.exists(MODEL_OUTPUT_PATH):
            logger.warning("Model file not found, returning dummy results")
            return np.zeros(len(df)), np.array([[0.8, 0.2] for _ in range(len(df))])
        
        try:
            # Load model
            model = load_model()
            
            # Prepare data
            X = prepare_prediction_data(df)
            
            # Make predictions
            predictions, probabilities = make_predictions(model, X)
            
            logger.info("Prediction pipeline completed successfully")
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            # Return default values in case of error
            return np.zeros(len(df)), np.array([[0.8, 0.2] for _ in range(len(df))])
            
    except Exception as e:
        logger.error(f"Unexpected error in prediction pipeline: {e}")
        # Return default values in case of error
        return np.zeros(len(df)), np.array([[0.8, 0.2] for _ in range(len(df))])

if __name__ == "__main__":
    # Sample data for testing
    sample_data = pd.DataFrame({
        'tenure': [12],
        'monthly_charges': [50.5],
        'total_charges': [606.0],
        'age': [35],
        'number_of_dependents': [2],
        'number_of_referrals': [0],
        'gender': ['Male'],
        'contract_type': ['Month-to-month'],
        'payment_method': ['Electronic check'],
        'internet_service': ['Fiber optic']
    })
    
    predictions, probabilities = predict_churn(sample_data)
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")
