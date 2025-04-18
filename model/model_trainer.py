"""
Module for training XGBoost models.
"""
import logging
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

def train_xgboost_model(X_train, y_train, params, eval_set=None):
    """
    Train an XGBoost model for customer churn prediction.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training target
    params : dict
        Hyperparameters for the model
    eval_set : list, optional
        Evaluation set for early stopping
        
    Returns:
    --------
    xgboost.XGBClassifier
        Trained XGBoost model
    """
    try:
        logger.info("Training XGBoost model")
        
        # Create model with parameters
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            gamma=params.get('gamma', 0),
            min_child_weight=params.get('min_child_weight', 1),
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Set up early stopping if eval_set is provided
        if eval_set is not None:
            model.fit(
                X_train, 
                y_train, 
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=True
            )
        else:
            model.fit(X_train, y_train)
        
        # Make predictions on training data
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        
        logger.info(f"Model training complete. Training accuracy: {accuracy:.4f}")
        return model
        
    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}")
        raise

def train_lightgbm_model(X_train, y_train, params, eval_set=None):
    """
    Train a LightGBM model for customer churn prediction.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training target
    params : dict
        Hyperparameters for the model
    eval_set : list, optional
        Evaluation set for early stopping
        
    Returns:
    --------
    lightgbm.LGBMClassifier
        Trained LightGBM model
    """
    try:
        logger.info("Training LightGBM model")
        
        # Import lightgbm here to avoid dependency issues
        import lightgbm as lgb
        
        # Create model with parameters
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', -1),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            min_child_samples=params.get('min_child_samples', 20),
            random_state=42
        )
        
        # Set up early stopping if eval_set is provided
        if eval_set is not None:
            model.fit(
                X_train, 
                y_train, 
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=True
            )
        else:
            model.fit(X_train, y_train)
        
        # Make predictions on training data
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        
        logger.info(f"Model training complete. Training accuracy: {accuracy:.4f}")
        return model
        
    except Exception as e:
        logger.error(f"Error training LightGBM model: {e}")
        raise
