"""
Feature engineering manager with improved handling of categorical data.
"""
import pandas as pd
import numpy as np
from app.ml.enhanced.feature_pipeline import EnhancedFeatureEngineeringPipeline
from app.ml.enhanced.options_features import (
    OptionsFeatureExtractor,
    OptionSpreadFeatureExtractor,
    MarketContextFeatureExtractor
)

class FeatureEngineeringManager:
    """
    Manager class for feature engineering operations.
    Provides a high-level interface for using the enhanced feature engineering pipeline.
    """
    def __init__(self, config=None):
        """
        Initialize the feature engineering manager.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with feature engineering settings
        """
        self.config = config or self._default_config()
        self._initialize_pipeline()
        
    def _default_config(self):
        """
        Create default configuration for feature engineering.
        
        Returns:
        --------
        dict
            Default configuration dictionary
        """
        return {
            'numerical_features': [
                'strike', 'bid', 'ask', 'underlyingPrice', 'daysToExpiration',
                'delta', 'gamma', 'theta', 'vega', 'rho', 'impliedVolatility',
                'volume', 'openInterest'
            ],
            'categorical_features': ['putCall', 'symbol'],  # Added 'symbol' as categorical
            'date_features': ['expirationDate'],
            'normalization': 'standard',
            'handle_outliers': True,
            'include_options_features': True,
            'include_spread_features': True,
            'include_market_features': True,
            'selection_method': 'k_best',
            'k': 30,
            'use_pca': False,
            'n_components': 0.95
        }
    
    def _initialize_pipeline(self):
        """
        Initialize the feature engineering pipeline based on configuration.
        """
        self.pipeline = EnhancedFeatureEngineeringPipeline(
            numerical_features=self.config.get('numerical_features'),
            categorical_features=self.config.get('categorical_features'),
            date_features=self.config.get('date_features'),
            normalization=self.config.get('normalization'),
            handle_outliers=self.config.get('handle_outliers'),
            include_options_features=self.config.get('include_options_features'),
            include_spread_features=self.config.get('include_spread_features'),
            include_market_features=self.config.get('include_market_features'),
            selection_method=self.config.get('selection_method'),
            k=self.config.get('k'),
            use_pca=self.config.get('use_pca'),
            n_components=self.config.get('n_components')
        )
    
    def process_options_data(self, options_data, target=None, fit=True):
        """
        Process options data through the feature engineering pipeline.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data to process
        target : array-like, optional
            Target values for supervised feature selection
        fit : bool
            Whether to fit the pipeline to the data
            
        Returns:
        --------
        pandas.DataFrame
            Processed data with engineered features
        """
        # Ensure data is in the right format
        if not isinstance(options_data, pd.DataFrame):
            options_data = pd.DataFrame(options_data)
        
        # Handle categorical features - ensure they're properly encoded
        # This is a preprocessing step to avoid the "could not convert string to float" error
        for col in self.config.get('categorical_features', []):
            if col in options_data.columns:
                # Convert to category type to ensure proper handling
                options_data[col] = options_data[col].astype('category')
        
        # Apply the pipeline
        if fit and target is not None:
            result = self.pipeline.fit_transform(options_data, target)
        elif fit:
            result = self.pipeline.fit_transform(options_data)
        else:
            result = self.pipeline.transform(options_data)
        
        # Convert to DataFrame if not already
        if not isinstance(result, pd.DataFrame):
            # Try to get feature names
            feature_names = self.pipeline.get_feature_names()
            if feature_names is not None:
                result = pd.DataFrame(result, columns=feature_names)
            else:
                result = pd.DataFrame(result)
        
        return result
    
    def get_feature_importance(self, X, y):
        """
        Calculate feature importance using multiple methods.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        y : array-like
            Target values
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importance scores
        """
        # Process data through pipeline
        X_processed = self.process_options_data(X, y)
        
        # Initialize importance DataFrame
        importance_df = pd.DataFrame()
        
        # Calculate correlation-based importance
        try:
            corr = X_processed.corrwith(pd.Series(y)).abs()
            importance_df['correlation'] = corr
        except Exception as e:
            print(f"Error calculating correlation importance: {e}")
        
        # Calculate mutual information importance
        try:
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(X_processed, y)
            mi_series = pd.Series(mi, index=X_processed.columns)
            importance_df['mutual_info'] = mi_series
        except Exception as e:
            print(f"Error calculating mutual information importance: {e}")
        
        # Calculate permutation importance if possible
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance
            
            # Train a simple model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_processed, y)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X_processed, y, n_repeats=5, random_state=42)
            perm_series = pd.Series(perm_importance.importances_mean, index=X_processed.columns)
            importance_df['permutation'] = perm_series
        except Exception as e:
            print(f"Error calculating permutation importance: {e}")
        
        # Normalize and calculate combined importance
        if not importance_df.empty:
            # Normalize each column
            for col in importance_df.columns:
                if importance_df[col].sum() > 0:
                    importance_df[col] = importance_df[col] / importance_df[col].sum()
            
            # Calculate combined importance
            importance_df['combined'] = importance_df.mean(axis=1)
            
            # Sort by combined importance
            importance_df = importance_df.sort_values('combined', ascending=False)
        
        return importance_df
    
    def get_top_features(self, X, y, n=10):
        """
        Get the top n most important features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        y : array-like
            Target values
        n : int
            Number of top features to return
            
        Returns:
        --------
        list
            List of top feature names
        """
        importance = self.get_feature_importance(X, y)
        if 'combined' in importance.columns:
            return importance.nlargest(n, 'combined').index.tolist()
        return []
    
    def create_feature_report(self, X, y):
        """
        Create a comprehensive report on feature engineering results.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        y : array-like
            Target values
            
        Returns:
        --------
        dict
            Report dictionary with feature statistics and importance
        """
        # Process data
        X_processed = self.process_options_data(X, y)
        
        # Calculate feature importance
        importance = self.get_feature_importance(X, y)
        
        # Calculate feature statistics
        stats = X_processed.describe().T
        
        # Calculate correlations between features
        corr_matrix = X_processed.corr()
        
        # Find highly correlated features
        high_corr = {}
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.8:
                    high_corr[f"{col1}_{col2}"] = corr_val
        
        # Compile report
        report = {
            'feature_count': len(X_processed.columns),
            'top_features': self.get_top_features(X, y),
            'feature_importance': importance.to_dict(),
            'feature_stats': stats.to_dict(),
            'high_correlations': high_corr
        }
        
        return report
