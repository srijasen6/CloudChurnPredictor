"""
Flask application for the churn prediction dashboard and API.
"""
import os
import json
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
from pipelines.prediction_pipeline import predict_churn
from pipelines.extraction_pipeline import extract_data
from pipelines.training_pipeline import train_model
from config import MODEL_METRICS_PATH, FEATURE_IMPORTANCE_PATH, MODEL_OUTPUT_PATH
import pickle

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load model and metadata if it exists
def load_model():
    if os.path.exists(MODEL_OUTPUT_PATH):
        try:
            with open(MODEL_OUTPUT_PATH, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    return None

@app.route('/')
def index():
    """
    Render the main dashboard page.
    """
    # Check if model exists
    model_exists = os.path.exists(MODEL_OUTPUT_PATH)
    
    # Load model metrics if available
    metrics = {}
    if os.path.exists(MODEL_METRICS_PATH):
        try:
            with open(MODEL_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading model metrics: {e}")
    
    # Load feature importance if available
    feature_importance = {}
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        try:
            with open(FEATURE_IMPORTANCE_PATH, 'r') as f:
                feature_importance = json.load(f)
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
    
    return render_template('index.html', 
                           model_exists=model_exists,
                           metrics=metrics,
                           feature_importance=feature_importance)

@app.route('/train', methods=['POST'])
def train():
    """
    Endpoint to trigger the training pipeline.
    """
    try:
        # Run the training pipeline
        result = train_model()
        if result.startswith("Error"):
            flash(f"{result}", "warning")
        else:
            flash(f"Training completed successfully", "success")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        flash(f"Error in training pipeline: {str(e)}", "danger")
    
    return redirect(url_for('index'))

@app.route('/extract', methods=['POST'])
def extract():
    """
    Endpoint to trigger the data extraction pipeline.
    """
    try:
        # Run the extraction pipeline
        result = extract_data()
        if result.startswith("Error"):
            flash(f"{result}", "warning")
        else:
            flash(f"Data extraction completed successfully", "success")
    except Exception as e:
        logger.error(f"Error in extraction pipeline: {e}")
        flash(f"Error in extraction pipeline: {str(e)}", "danger")
    
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Page for making individual predictions.
    """
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'tenure': float(request.form.get('tenure', 0)),
                'monthly_charges': float(request.form.get('monthly_charges', 0)),
                'total_charges': float(request.form.get('total_charges', 0)),
                'age': float(request.form.get('age', 0)),
                'number_of_dependents': float(request.form.get('number_of_dependents', 0)),
                'number_of_referrals': float(request.form.get('number_of_referrals', 0)),
                'gender': request.form.get('gender', ''),
                'contract_type': request.form.get('contract_type', ''),
                'payment_method': request.form.get('payment_method', ''),
                'internet_service': request.form.get('internet_service', '')
            }
            
            # Convert to DataFrame 
            df = pd.DataFrame([data])
            
            # Run prediction
            prediction, probability = predict_churn(df)
            
            return render_template('prediction.html', 
                                  prediction=prediction[0], 
                                  probability=round(probability[0][1] * 100, 2),
                                  customer_data=data)
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            flash(f"Error making prediction: {str(e)}", "danger")
            return render_template('prediction.html')
    
    return render_template('prediction.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for batch prediction.
    """
    try:
        # Get JSON data from request
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions, probabilities = predict_churn(df)
        
        # Prepare response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "prediction": int(pred),
                "churn_probability": float(probabilities[i][1]),
                "retain_probability": float(probabilities[i][0])
            })
            
        return jsonify({"results": results})
    
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
