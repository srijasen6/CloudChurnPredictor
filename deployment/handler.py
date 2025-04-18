"""
AWS Lambda handler for serverless deployment of the churn prediction model.
"""
import json
import logging
import os
import pickle
import boto3
import numpy as np
import pandas as pd
from io import BytesIO

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

# Environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'churn-prediction-models')
MODEL_KEY = os.environ.get('MODEL_KEY', 'model.pkl')
PREPROCESSOR_KEY = os.environ.get('PREPROCESSOR_KEY', 'preprocessor.pkl')

# Global variables to store model and preprocessor
model = None
preprocessor = None

def load_from_s3(bucket, key):
    """
    Load a pickle file from S3.
    
    Parameters:
    -----------
    bucket : str
        S3 bucket name
    key : str
        S3 object key
        
    Returns:
    --------
    object
        Unpickled object
    """
    try:
        logger.info(f"Loading {key} from S3 bucket {bucket}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response['Body'].read()
        obj = pickle.loads(body)
        logger.info(f"Successfully loaded {key}")
        return obj
    except Exception as e:
        logger.error(f"Error loading {key} from S3: {e}")
        raise

def load_model_and_preprocessor():
    """
    Load model and preprocessor from S3.
    
    Returns:
    --------
    tuple
        (model, preprocessor)
    """
    global model, preprocessor
    
    if model is None:
        model = load_from_s3(MODEL_BUCKET, MODEL_KEY)
    
    if preprocessor is None:
        preprocessor = load_from_s3(MODEL_BUCKET, PREPROCESSOR_KEY)
    
    return model, preprocessor

def prepare_data(data):
    """
    Prepare data for prediction using the preprocessor.
    
    Parameters:
    -----------
    data : list of dict
        Input data
        
    Returns:
    --------
    numpy.ndarray
        Processed features
    """
    # Load preprocessor if not already loaded
    _, preprocessor = load_model_and_preprocessor()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Transform data
    X = preprocessor.transform(df)
    
    return X

def predict(event, context):
    """
    Lambda handler for making a single prediction.
    
    Parameters:
    -----------
    event : dict
        Lambda event
    context : object
        Lambda context
        
    Returns:
    --------
    dict
        API response
    """
    try:
        logger.info("Received prediction request")
        
        # Parse input data
        body = json.loads(event.get('body', '{}'))
        
        # Ensure input is a dictionary
        if not isinstance(body, dict):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Input must be a JSON object'})
            }
        
        # Convert to list for processing
        data = [body]
        
        # Prepare data
        X = prepare_data(data)
        
        # Load model if not already loaded
        model, _ = load_model_and_preprocessor()
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].tolist()
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'churn_probability': probability[1],
            'retain_probability': probability[0]
        }
        
        logger.info(f"Prediction result: {response}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(response),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }

def batch_predict(event, context):
    """
    Lambda handler for making batch predictions.
    
    Parameters:
    -----------
    event : dict
        Lambda event
    context : object
        Lambda context
        
    Returns:
    --------
    dict
        API response
    """
    try:
        logger.info("Received batch prediction request")
        
        # Parse input data
        body = json.loads(event.get('body', '{}'))
        
        # Ensure input is a list
        if not isinstance(body, list):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Input must be a JSON array of objects'})
            }
        
        # Prepare data
        X = prepare_data(body)
        
        # Load model if not already loaded
        model, _ = load_model_and_preprocessor()
        
        # Make predictions
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X).tolist()
        
        # Prepare response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'prediction': int(pred),
                'churn_probability': probabilities[i][1],
                'retain_probability': probabilities[i][0]
            })
        
        logger.info(f"Batch prediction complete, processed {len(results)} records")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'results': results}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
        
    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }

def info(event, context):
    """
    Lambda handler for getting model information.
    
    Parameters:
    -----------
    event : dict
        Lambda event
    context : object
        Lambda context
        
    Returns:
    --------
    dict
        API response
    """
    try:
        logger.info("Received model info request")
        
        # Load model if not already loaded
        model, preprocessor = load_model_and_preprocessor()
        
        # Get model info
        model_info = {
            'model_type': type(model).__name__,
            'required_features': [c for c_list in preprocessor.transformers_
                                 for c in c_list[2] if c_list[2] is not None]
        }
        
        # Add XGBoost specific info if available
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            model_info['num_trees'] = booster.num_boosted_rounds()
        
        logger.info("Model info request processed successfully")
        
        return {
            'statusCode': 200,
            'body': json.dumps(model_info),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
