"""
Enhanced Feature Engineering Pipeline for Options Trading ML Models.
Integrates specialized options feature extractors with scikit-learn pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

# Import custom feature extractors
from app.ml.enhanced.options_features import (
    OptionsFeatureExtractor,
    OptionSpreadFeatureExtractor,
    MarketContextFeatureExtractor
)

class EnhancedFeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    Enhanced feature engineering pipeline for options data.
    Combines preprocessing, feature extraction, and feature selection.
    """
    def __init__(self, numerical_features=None, categorical_features=None, 
                 date_features=None, normalization='standard', handle_outliers=True,
                 include_options_features=True, include_spread_features=True,
                 include_market_features=True, selection_method='k_best', k=20, 
                 use_pca=False, n_components=0.95):
        """
        Initialize the enhanced feature engineering pipeline.
        
        Parameters:
        -----------
        numerical_features : list
            List of numerical feature column names
        categorical_features : list
            List of categorical feature column names
        date_features : list
            List of date/time feature column names
        normalization : str
            Type of normalization to apply ('standard', 'minmax', or None)
        handle_outliers : bool
            Whether to apply outlier handling
        include_options_features : bool
            Whether to include options-specific features
        include_spread_features : bool
            Whether to include option spread features
        include_market_features : bool
            Whether to include market context features
        selection_method : str
            Feature selection method ('k_best', 'mutual_info', 'correlation')
        k : int
            Number of top features to select
        use_pca : bool
            Whether to apply PCA after feature selection
        n_components : float or int
            Number of components to keep if using PCA
            If float, represents the variance to be explained (0.0 to 1.0)
            If int, represents the exact number of components
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.date_features = date_features
        self.normalization = normalization
        self.handle_outliers = handle_outliers
        self.include_options_features = include_options_features
        self.include_spread_features = include_spread_features
        self.include_market_features = include_market_features
        self.selection_method = selection_method
        self.k = k
        self.use_pca = use_pca
        self.n_components = n_components
        
        # Initialize pipeline components
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """
        Initialize the pipeline components based on configuration.
        """
        # Create preprocessing steps
        preprocessing_steps = []
        
        # Numerical features preprocessing
        if self.numerical_features:
            num_transformer = self._create_numerical_transformer()
            preprocessing_steps.append(
                ('numerical', num_transformer, self.numerical_features)
            )
        
        # Categorical features preprocessing
        if self.categorical_features:
            cat_transformer = self._create_categorical_transformer()
            preprocessing_steps.append(
                ('categorical', cat_transformer, self.categorical_features)
            )
        
        # Date features preprocessing
        if self.date_features:
            date_transformer = self._create_date_transformer()
            preprocessing_steps.append(
                ('date', date_transformer, self.date_features)
            )
        
        # Create the column transformer for preprocessing
        self.preprocessor = ColumnTransformer(
            transformers=preprocessing_steps,
            remainder='passthrough'
        )
        
        # Create feature extraction steps
        feature_extraction_steps = []
        
        # Options features extraction
        if self.include_options_features:
            options_extractor = OptionsFeatureExtractor(
                include_greeks=True,
                include_iv_features=True,
                include_term_structure=True,
                include_skew=True,
                include_moneyness=True,
                include_time_features=True
            )
            feature_extraction_steps.append(
                ('options_features', options_extractor)
            )
        
        # Spread features extraction
        if self.include_spread_features:
            spread_extractor = OptionSpreadFeatureExtractor(
                include_bid_ask=True,
                include_volume=True,
                include_open_interest=True
            )
            feature_extraction_steps.append(
                ('spread_features', spread_extractor)
            )
        
        # Market context features extraction
        if self.include_market_features:
            market_extractor = MarketContextFeatureExtractor(
                include_market_indicators=True,
                include_sector_data=True,
                include_earnings=True,
                include_dividends=True
            )
            feature_extraction_steps.append(
                ('market_features', market_extractor)
            )
        
        # Create feature union for extraction if multiple extractors
        if len(feature_extraction_steps) > 1:
            self.feature_extractor = FeatureUnion(
                transformer_list=feature_extraction_steps
            )
        elif len(feature_extraction_steps) == 1:
            self.feature_extractor = feature_extraction_steps[0][1]
        else:
            self.feature_extractor = None
        
        # Create feature selection component
        self.feature_selector = self._create_feature_selector()
        
        # Create the complete pipeline
        pipeline_steps = [('preprocessor', self.preprocessor)]
        
        if self.feature_extractor:
            pipeline_steps.append(('feature_extractor', self.feature_extractor))
        
        if self.feature_selector:
            pipeline_steps.append(('feature_selector', self.feature_selector))
        
        self.pipeline = Pipeline(pipeline_steps)
    
    def _create_numerical_transformer(self):
        """
        Create transformer for numerical features.
        
        Returns:
        --------
        Pipeline
            Scikit-learn pipeline for numerical feature preprocessing
        """
        steps = []
        
        # Add imputer for missing values
        steps.append(('imputer', SimpleImputer(strategy='median')))
        
        # Add outlier handling if enabled
        if self.handle_outliers:
            steps.append(('outlier_handler', OutlierHandler()))
        
        # Add normalization if specified
        if self.normalization == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif self.normalization == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        
        return Pipeline(steps)
    
    def _create_categorical_transformer(self):
        """
        Create transformer for categorical features.
        
        Returns:
        --------
        Pipeline
            Scikit-learn pipeline for categorical feature preprocessing
        """
        return Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
    
    def _create_date_transformer(self):
        """
        Create transformer for date features.
        
        Returns:
        --------
        Pipeline
            Scikit-learn pipeline for date feature preprocessing
        """
        return Pipeline([
            ('date_encoder', DateEncoder())
        ])
    
    def _create_feature_selector(self):
        """
        Create feature selector based on configuration.
        
        Returns:
        --------
        BaseEstimator
            Scikit-learn compatible feature selector
        """
        if self.selection_method == 'k_best':
            selector = SelectKBest(f_regression, k=self.k)
        elif self.selection_method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=self.k)
        elif self.selection_method == 'correlation':
            selector = CorrelationSelector(k=self.k)
        else:
            return None
        
        # Add PCA if enabled
        if self.use_pca:
            return Pipeline([
                ('selector', selector),
                ('pca', PCA(n_components=self.n_components))
            ])
        
        return selector
    
    def fit(self, X, y=None):
        """
        Fit the feature engineering pipeline.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        y : array-like, optional
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        """
        Transform the data using the feature engineering pipeline.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            Transformed data
        """
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data using the feature engineering pipeline.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        y : array-like, optional
            Target values
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            Transformed data
        """
        return self.pipeline.fit_transform(X, y)
    
    def get_feature_names(self):
        """
        Get the feature names after transformation.
        
        Returns:
        --------
        list
            List of feature names
        """
        try:
            # Try to get feature names from pipeline
            if hasattr(self.pipeline, 'get_feature_names_out'):
                return self.pipeline.get_feature_names_out()
            
            # If not available, try to reconstruct from components
            feature_names = []
            
            # Add preprocessed features
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names.extend(self.preprocessor.get_feature_names_out())
            
            # Add extracted features
            if self.feature_extractor and hasattr(self.feature_extractor, 'get_feature_names_out'):
                feature_names.extend(self.feature_extractor.get_feature_names_out())
            
            return feature_names
        except Exception:
            return None


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handles outliers in numerical data.
    """
    def __init__(self, method='clip', threshold=3.0):
        """
        Initialize the outlier handler.
        
        Parameters:
        -----------
        method : str
            Method to handle outliers ('clip', 'remove', or 'winsorize')
        threshold : float
            Threshold for outlier detection (in standard deviations)
        """
        self.method = method
        self.threshold = threshold
        self.lower_bounds = None
        self.upper_bounds = None
    
    def fit(self, X, y=None):
        """
        Fit the outlier handler.
        
        Parameters:
        -----------
        X : array-like
            Input data
        y : array-like, optional
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Calculate bounds for each feature
        self.lower_bounds = np.mean(X, axis=0) - self.threshold * np.std(X, axis=0)
        self.upper_bounds = np.mean(X, axis=0) + self.threshold * np.std(X, axis=0)
        
        return self
    
    def transform(self, X):
        """
        Transform the data by handling outliers.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        array-like
            Data with outliers handled
        """
        if self.lower_bounds is None or self.upper_bounds is None:
            return X
        
        # Make a copy to avoid modifying the original
        X_transformed = X.copy()
        
        if self.method == 'clip':
            # Clip values outside bounds
            for i in range(X.shape[1]):
                X_transformed[:, i] = np.clip(
                    X_transformed[:, i], 
                    self.lower_bounds[i], 
                    self.upper_bounds[i]
                )
        elif self.method == 'winsorize':
            # Replace outliers with bounds
            for i in range(X.shape[1]):
                X_transformed[:, i] = np.where(
                    X_transformed[:, i] < self.lower_bounds[i],
                    self.lower_bounds[i],
                    np.where(
                        X_transformed[:, i] > self.upper_bounds[i],
                        self.upper_bounds[i],
                        X_transformed[:, i]
                    )
                )
        
        return X_transformed


class DateEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes date features into numerical representations.
    """
    def __init__(self, include_year=True, include_month=True, include_day=True,
                 include_weekday=True, include_quarter=True, include_is_month_end=True):
        """
        Initialize the date encoder.
        
        Parameters:
        -----------
        include_year : bool
            Whether to include year as a feature
        include_month : bool
            Whether to include month as a feature
        include_day : bool
            Whether to include day as a feature
        include_weekday : bool
            Whether to include weekday as a feature
        include_quarter : bool
            Whether to include quarter as a feature
        include_is_month_end : bool
            Whether to include is_month_end as a feature
        """
        self.include_year = include_year
        self.include_month = include_month
        self.include_day = include_day
        self.include_weekday = include_weekday
        self.include_quarter = include_quarter
        self.include_is_month_end = include_is_month_end
    
    def fit(self, X, y=None):
        """
        Fit the date encoder (no-op).
        
        Parameters:
        -----------
        X : array-like
            Input data
        y : array-like, optional
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        return self
    
    def transform(self, X):
        """
        Transform date features into numerical representations.
        
        Parameters:
        -----------
        X : array-like
            Input data with date features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with encoded date features
        """
        # Convert to DataFrame if not already
        X_df = pd.DataFrame(X)
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=X_df.index)
        
        # Process each column
        for col in X_df.columns:
            # Convert to datetime if not already
            try:
                dates = pd.to_datetime(X_df[col])
                
                # Extract date components
                if self.include_year:
                    result[f'{col}_year'] = dates.dt.year
                
                if self.include_month:
                    result[f'{col}_month'] = dates.dt.month
                
                if self.include_day:
                    result[f'{col}_day'] = dates.dt.day
                
                if self.include_weekday:
                    result[f'{col}_weekday'] = dates.dt.weekday
                
                if self.include_quarter:
                    result[f'{col}_quarter'] = dates.dt.quarter
                
                if self.include_is_month_end:
                    result[f'{col}_is_month_end'] = dates.dt.is_month_end.astype(int)
                
                # Add days to specific events if available
                if 'days_to_expiration' in X_df.columns:
                    result[f'{col}_days_to_expiration'] = X_df['days_to_expiration']
                
                if 'days_to_earnings' in X_df.columns:
                    result[f'{col}_days_to_earnings'] = X_df['days_to_earnings']
                
            except Exception:
                # If conversion fails, keep original column
                result[col] = X_df[col]
        
        return result


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Selects features based on correlation with target.
    """
    def __init__(self, k=10):
        """
        Initialize the correlation selector.
        
        Parameters:
        -----------
        k : int
            Number of top features to select
        """
        self.k = k
        self.selected_features = None
    
    def fit(self, X, y):
        """
        Fit the correlation selector.
        
        Parameters:
        -----------
        X : array-like
            Input data
        y : array-like
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Calculate correlation with target
        correlations = np.abs(np.array([np.corrcoef(X.iloc[:, i], y)[0, 1] for i in range(X.shape[1])]))
        
        # Handle NaN values
        correlations = np.nan_to_num(correlations)
        
        # Select top k features
        self.selected_features = np.argsort(correlations)[-self.k:]
        
        return self
    
    def transform(self, X):
        """
        Transform the data by selecting top correlated features.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        array-like
            Data with selected features
        """
        if self.selected_features is None:
            return X
        
        # Convert to numpy array if not already
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return X[:, self.selected_features]
