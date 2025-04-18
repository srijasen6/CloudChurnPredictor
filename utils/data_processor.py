"""
Data processing utilities for the churn prediction pipeline.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for processing and transforming data for the churn prediction model.
    """
    def __init__(self, 
                cat_features=CATEGORICAL_FEATURES, 
                num_features=NUMERICAL_FEATURES,
                target=TARGET_COLUMN):
        """
        Initialize data processor.
        
        Parameters:
        -----------
        cat_features : list
            List of categorical feature names
        num_features : list
            List of numerical feature names
        target : str
            Target column name
        """
        self.categorical_features = cat_features
        self.numerical_features = num_features
        self.target = target
        self.preprocessor = None
    
    def create_preprocessor(self):
        """
        Create a column transformer for preprocessing data.
        
        Returns:
        --------
        sklearn.compose.ColumnTransformer
            Preprocessor for transforming features
        """
        try:
            logger.info("Creating data preprocessor")
            
            # Numerical features pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Categorical features pipeline
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Column transformer
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, self.numerical_features),
                ('cat', cat_pipeline, self.categorical_features)
            ])
            
            self.preprocessor = preprocessor
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error creating preprocessor: {e}")
            raise
    
    def prepare_data(self, df, train=True, test_size=0.2, random_state=42):
        """
        Prepare data for model training or prediction.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        train : bool
            If True, split data into train/test sets
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary containing processed data
        """
        try:
            logger.info("Preparing data for model")
            
            # Check if required columns exist
            required_cols = self.categorical_features + self.numerical_features
            if train:
                required_cols.append(self.target)
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Create preprocessor if not already created
            if self.preprocessor is None:
                self.create_preprocessor()
            
            if train:
                # Split data into features and target
                X = df[self.categorical_features + self.numerical_features]
                y = df[self.target].astype(int)
                
                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Fit preprocessor on training data
                X_train_processed = self.preprocessor.fit_transform(X_train)
                X_test_processed = self.preprocessor.transform(X_test)
                
                return {
                    'X_train': X_train_processed,
                    'X_test': X_test_processed,
                    'y_train': y_train,
                    'y_test': y_test,
                    'feature_names': self.get_feature_names()
                }
            else:
                # Only transform the data for prediction
                X = df[self.categorical_features + self.numerical_features]
                X_processed = self.preprocessor.transform(X)
                
                return {
                    'X': X_processed,
                    'feature_names': self.get_feature_names()
                }
                
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def get_feature_names(self):
        """
        Get feature names after preprocessing.
        
        Returns:
        --------
        list
            List of feature names after preprocessing
        """
        # Extract feature names from preprocessor
        num_features = self.numerical_features
        
        # Get categorical feature names after one-hot encoding
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = []
        for i, feature in enumerate(self.categorical_features):
            cat_values = cat_encoder.categories_[i]
            for value in cat_values:
                cat_features.append(f"{feature}_{value}")
        
        return num_features + cat_features
    
    def save_preprocessor(self, path='./model/preprocessor.pkl'):
        """
        Save preprocessor to disk.
        
        Parameters:
        -----------
        path : str
            Path to save the preprocessor
        """
        try:
            logger.info(f"Saving preprocessor to {path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save preprocessor
            with open(path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
                
            logger.info("Preprocessor saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
            raise
    
    def load_preprocessor(self, path='./model/preprocessor.pkl'):
        """
        Load preprocessor from disk.
        
        Parameters:
        -----------
        path : str
            Path to load the preprocessor from
        """
        try:
            logger.info(f"Loading preprocessor from {path}")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Preprocessor file not found: {path}")
            
            # Load preprocessor
            with open(path, 'rb') as f:
                self.preprocessor = pickle.load(f)
                
            logger.info("Preprocessor loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise
