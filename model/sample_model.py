"""
Script to create a simple sample model for testing.
"""
import os
import pickle
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create model directory if it doesn't exist
os.makedirs('./model', exist_ok=True)

# Create a simple random forest model
model = RandomForestClassifier(n_estimators=10, max_depth=3)

# Create some sample data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Train the model
model.fit(X, y)

# Save the model
with open('./model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Sample model saved to './model/model.pkl'")

# Create sample metrics
metrics = {
    'accuracy': 0.85,
    'precision': 0.82,
    'recall': 0.78,
    'f1': 0.80,
    'auc_roc': 0.88,
    'confusion_matrix': [[45, 5], [10, 40]]
}

# Save metrics
with open('./model/metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Sample metrics saved to './model/metrics.json'")

# Create sample feature importance
feature_importance = {
    'tenure': 0.25,
    'monthly_charges': 0.20,
    'total_charges': 0.18,
    'contract_type_Month-to-month': 0.15,
    'payment_method_Electronic check': 0.10,
    'internet_service_Fiber optic': 0.08,
    'age': 0.04
}

# Save feature importance
with open('./model/feature_importance.json', 'w') as f:
    json.dump(feature_importance, f)

print("Sample feature importance saved to './model/feature_importance.json'")

# Create a sample preprocessor (simple identity transform)
class SamplePreprocessor:
    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        return X

preprocessor = SamplePreprocessor()

# Save preprocessor
with open('./model/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("Sample preprocessor saved to './model/preprocessor.pkl'")