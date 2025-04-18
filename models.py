"""
Database models for the application.
"""
from app import db
from datetime import datetime

class PredictionLog(db.Model):
    """
    Logs of predictions made through the API or UI.
    """
    id = db.Column(db.Integer, primary_key=True)
    input_data = db.Column(db.JSON)
    prediction = db.Column(db.Boolean)
    probability = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PredictionLog {self.id}>'

class ModelMetrics(db.Model):
    """
    Stores metrics for each trained model version.
    """
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50))
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    auc_roc = db.Column(db.Float)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    hyperparameters = db.Column(db.JSON)
    
    def __repr__(self):
        return f'<ModelMetrics {self.version}>'

class DataExtractionLog(db.Model):
    """
    Logs details of each data extraction run.
    """
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(50))  # 'aws' or 'gcp'
    rows_extracted = db.Column(db.Integer)
    status = db.Column(db.String(50))  # 'success' or 'failed'
    error_message = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<DataExtractionLog {self.id} - {self.source}>'
