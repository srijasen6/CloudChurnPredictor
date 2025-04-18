"""
Feature engineering utilities for the churn prediction pipeline.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Class for feature engineering on customer data.
    """

    def __init__(self):
        """
        Initialize feature engineer.
        """
        logger.info("Initializing feature engineer")

    def engineer_features(self, df):
        """
        Apply feature engineering to the input DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        try:
            logger.info("Applying feature engineering")

            # Create a copy to avoid modifying the original DataFrame
            result = df.copy()

            # --- Add feature engineering steps here ---

            # Example: Calculate lifetime value if monthly_charges and tenure are available
            if 'monthly_charges' in result.columns and 'tenure' in result.columns:
                result['customer_lifetime_value'] = result[
                    'monthly_charges'] * result['tenure']
                logger.info("Added customer_lifetime_value feature")

            # Example: Calculate average monthly spending if total_charges and tenure are available
            if 'total_charges' in result.columns and 'tenure' in result.columns:
                # Avoid division by zero
                result['average_monthly_spending'] = result[
                    'total_charges'] / result['tenure'].replace(0, 1)
                logger.info("Added average_monthly_spending feature")

            # Example: Create ratio features for dependents
            if 'number_of_dependents' in result.columns and 'monthly_charges' in result.columns:
                result['charges_per_dependent'] = result['monthly_charges'] / (
                    result['number_of_dependents'] + 1)
                logger.info("Added charges_per_dependent feature")

            # Example: Create binary features
            if 'contract_type' in result.columns:
                result['is_month_to_month'] = (
                    result['contract_type'] == 'Month-to-month').astype(int)
                logger.info("Added is_month_to_month feature")

            # Example: Create interaction features
            if 'tenure' in result.columns and 'monthly_charges' in result.columns:
                result['tenure_charges_interaction'] = result[
                    'tenure'] * result['monthly_charges']
                logger.info("Added tenure_charges_interaction feature")

            # Example: Log transformation for skewed features
            if 'total_charges' in result.columns:
                result['log_total_charges'] = np.log1p(result['total_charges'])
                logger.info("Added log_total_charges feature")

            logger.info(f"Feature engineering complete. Shape: {result.shape}")
            return result

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise


class RatioTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create ratio features.
    """

    def __init__(self, numerator, denominator, name=None, offset=1e-6):
        """
        Initialize ratio transformer.
        
        Parameters:
        -----------
        numerator : str
            Column name for numerator
        denominator : str
            Column name for denominator
        name : str, optional
            Name for the new feature
        offset : float
            Small value to avoid division by zero
        """
        self.numerator = numerator
        self.denominator = denominator
        self.name = name or f"{numerator}_to_{denominator}_ratio"
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform data by creating ratio feature.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Transformed data with ratio feature
        """
        X_ = X.copy()
        X_[self.name] = X_[self.numerator] / (X_[self.denominator] +
                                              self.offset)
        return X_


class MissingValueIndicator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create missing value indicators.
    """

    def __init__(self, columns=None, suffix='_missing'):
        """
        Initialize missing value indicator.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to check for missing values
        suffix : str
            Suffix for indicator column names
        """
        self.columns = columns
        self.suffix = suffix

    def fit(self, X, y=None):
        # If columns not specified, use all columns
        if self.columns is None:
            self.columns = X.columns
        return self

    def transform(self, X):
        """
        Transform data by creating missing value indicators.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Transformed data with missing value indicators
        """
        X_ = X.copy()
        for col in self.columns:
            X_[f"{col}{self.suffix}"] = X_[col].isna().astype(int)
        return X_
