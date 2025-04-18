"""
Module for optimizing model hyperparameters using Optuna.
"""
import logging
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from config import N_TRIALS, TIMEOUT, CROSS_VALIDATION_FOLDS

logger = logging.getLogger(__name__)

def optimize_hyperparameters(X, y, n_trials=N_TRIALS, timeout=TIMEOUT, cv=CROSS_VALIDATION_FOLDS):
    """
    Optimize hyperparameters for XGBoost model using Optuna.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Training features
    y : numpy.ndarray
        Training target
    n_trials : int
        Number of Optuna trials
    timeout : int
        Timeout in seconds
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Dictionary of best hyperparameters
    """
    try:
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        def objective(trial):
            """
            Objective function for Optuna.
            
            Parameters:
            -----------
            trial : optuna.Trial
                Optuna trial object
                
            Returns:
            --------
            float
                Cross-validation score
            """
            # Define hyperparameters to optimize
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            
            # Create model
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                use_label_encoder=False,
                **params
            )
            
            # Evaluate model with cross-validation
            score = cross_val_score(
                model, 
                X, 
                y, 
                cv=cv, 
                scoring='roc_auc', 
                n_jobs=-1
            ).mean()
            
            return score
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best AUC-ROC: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
        
    except Exception as e:
        logger.error(f"Error optimizing hyperparameters: {e}")
        # Return default parameters in case of error
        return {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1
        }

def optimize_lightgbm_hyperparameters(X, y, n_trials=N_TRIALS, timeout=TIMEOUT, cv=CROSS_VALIDATION_FOLDS):
    """
    Optimize hyperparameters for LightGBM model using Optuna.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Training features
    y : numpy.ndarray
        Training target
    n_trials : int
        Number of Optuna trials
    timeout : int
        Timeout in seconds
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Dictionary of best hyperparameters
    """
    try:
        logger.info(f"Starting LightGBM hyperparameter optimization with {n_trials} trials")
        
        # Import lightgbm here to avoid dependency issues
        import lightgbm as lgb
        
        def objective(trial):
            """
            Objective function for Optuna.
            
            Parameters:
            -----------
            trial : optuna.Trial
                Optuna trial object
                
            Returns:
            --------
            float
                Cross-validation score
            """
            # Define hyperparameters to optimize
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
            
            # Create model
            model = lgb.LGBMClassifier(
                objective='binary',
                random_state=42,
                **params
            )
            
            # Evaluate model with cross-validation
            score = cross_val_score(
                model, 
                X, 
                y, 
                cv=cv, 
                scoring='roc_auc', 
                n_jobs=-1
            ).mean()
            
            return score
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best AUC-ROC: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
        
    except Exception as e:
        logger.error(f"Error optimizing LightGBM hyperparameters: {e}")
        # Return default parameters in case of error
        return {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0
        }
