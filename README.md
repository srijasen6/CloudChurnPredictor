# Multi-Cloud Customer Churn Prediction Pipeline

A comprehensive end-to-end data pipeline for predicting customer churn using data from multiple cloud sources (AWS S3 and Google BigQuery). This application allows users to extract data, train machine learning models, and make predictions through both a web interface and API.

## Features

- **Multi-Cloud Data Integration**: Extract and merge data from AWS S3 and Google BigQuery
- **Automated Machine Learning**: Train XGBoost models with optimized hyperparameters
- **Feature Engineering**: Apply transformations to improve model performance
- **Web Dashboard**: Visualize model metrics and feature importance
- **Prediction API**: Make individual and batch predictions via REST API
- **Serverless Deployment**: Deploy models as serverless functions

![appdemo](https://github.com/user-attachments/assets/35a9ac6e-d118-41e5-9c26-a6df3e8a9d90)


## Project Structure

```
├── data/                     # Data storage directory
├── deployment/               # Serverless deployment files
│   ├── handler.py            # AWS Lambda handler
│   └── serverless.yml        # Serverless Framework config
├── model/                    # Model artifacts
│   ├── model.pkl             # Trained model
│   ├── metrics.json          # Model performance metrics
│   ├── feature_importance.json # Feature importance scores
│   ├── preprocessor.pkl      # Data preprocessor
│   ├── model_optimizer.py    # Hyperparameter optimization
│   ├── model_trainer.py      # Model training utilities
├── pipelines/                # Data and training pipelines
│   ├── extraction_pipeline.py # Data extraction and merging
│   ├── prediction_pipeline.py # Prediction utilities
│   └── training_pipeline.py  # Model training workflow
├── static/                   # CSS, JS, and other static files
├── templates/                # HTML templates
│   ├── index.html            # Dashboard template
│   ├── layout.html           # Base layout
│   └── prediction.html       # Prediction form
├── tests/                    # Unit and integration tests
├── utils/                    # Utility functions
│   ├── aws_connector.py      # AWS S3 connector
│   ├── gcp_connector.py      # Google BigQuery connector
│   ├── mock_connectors.py    # Mock connectors for testing
│   ├── data_processor.py     # Data preprocessing utilities
│   └── feature_engineering.py # Feature transformation utilities
├── app.py                    # Flask application
├── config.py                 # Configuration settings
├── main.py                   # Application entry point
└── README.md                 # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- AWS account (for production use)
- Google Cloud Platform account (for production use)


   ```

### Running the Application

1. Start the Flask application:
   ```
   python main.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Data Extraction

1. Navigate to the dashboard
2. Click the "Extract Data" button to pull and merge data from cloud sources
3. The data will be processed and saved to the `data` directory

### Model Training

1. Navigate to the dashboard
2. Click the "Train Model" button to train a new model
3. Model metrics and feature importance will be displayed on the dashboard

### Making Predictions

#### Single Prediction (Web Interface)

1. Navigate to the prediction page
2. Fill in the customer attributes
3. Click "Predict" to see the churn prediction and probability





## Acknowledgments

- XGBoost for the gradient boosting framework
- Flask for the web application framework
- Pandas and Scikit-learn for data processing
- AWS and Google Cloud Platform for cloud services

