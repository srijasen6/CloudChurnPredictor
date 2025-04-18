"""
Tests for ML model training and prediction.
"""
import unittest
import pandas as pd
import numpy as np
import os
import pickle
from unittest.mock import patch, MagicMock
from model.model_trainer import train_xgboost_model
from model.model_optimizer import optimize_hyperparameters

class TestModelTraining(unittest.TestCase):
    """
    Test cases for model training and optimization.
    """
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test data
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        self.X = X
        self.y = y
        
    def test_xgboost_model_training(self):
        """Test XGBoost model training."""
        # Arrange
        params = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1
        }
        
        # Act
        model = train_xgboost_model(self.X, self.y, params)
        
        # Assert
        self.assertIsNotNone(model)
        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.max_depth, 3)
        self.assertEqual(model.learning_rate, 0.1)
        
        # Test prediction
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
    @patch('optuna.create_study')
    def test_hyperparameter_optimization(self, mock_create_study):
        """Test hyperparameter optimization with Optuna."""
        # Arrange
        mock_study = MagicMock()
        mock_study.best_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1
        }
        mock_study.best_value = 0.95
        mock_study.best_trial = MagicMock()
        mock_study.best_trial.number = 42
        mock_create_study.return_value = mock_study
        
        # Act
        with patch('sklearn.model_selection.cross_val_score', return_value=np.array([0.95])):
            best_params = optimize_hyperparameters(self.X, self.y, n_trials=1)
        
        # Assert
        mock_create_study.assert_called_once()
        self.assertEqual(best_params['n_estimators'], 100)
        self.assertEqual(best_params['max_depth'], 5)
        self.assertEqual(best_params['learning_rate'], 0.1)
        
    def test_model_serialization(self):
        """Test model serialization and deserialization."""
        # Arrange
        params = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1
        }
        model = train_xgboost_model(self.X, self.y, params)
        
        # Act - Save model
        with open('test_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Act - Load model
        with open('test_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Assert
        self.assertEqual(model.n_estimators, loaded_model.n_estimators)
        self.assertEqual(model.max_depth, loaded_model.max_depth)
        
        # Clean up
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')

if __name__ == '__main__':
    unittest.main()
