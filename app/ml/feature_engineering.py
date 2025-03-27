"""
Feature Engineering Pipeline for Options Trading ML Models.
Transforms raw market and options data into features suitable for machine learning models.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesses raw market and options data for machine learning.
    Handles missing values, outliers, and normalization.
    """
    def __init__(self, numerical_features=None, categorical_features=None, 
                 date_features=None, normalization='standard', handle_outliers=True):
        """
        Initialize the preprocessor with feature specifications.
        
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
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.date_features = date_features
        self.normalization = normalization
        self.handle_outliers = handle_outliers
        self.preprocessor = None
        
    def fit(self, X, y=None):
        """
        Fit the preprocessor to the data.
        
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
        # Infer feature types if not provided
        if self.numerical_features is None:
            self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
        if self.date_features is None:
            # Try to identify date columns
            date_cols = []
            for col in X.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        pd.to_datetime(X[col])
                        date_cols.append(col)
                    except:
                        pass
            self.date_features = date_cols
        
        # Create preprocessing steps for each feature type
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler() if self.normalization == 'standard' else 
                      MinMaxScaler() if self.normalization == 'minmax' else None)
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create column transformer
        transformers = []
        
        if self.numerical_features:
            transformers.append(('num', numeric_transformer, self.numerical_features))
            
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))
            
        if self.date_features:
            # We'll handle date features separately in transform
            pass
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Transformed data
        """
        # Handle outliers if requested
        if self.handle_outliers:
            X = self._handle_outliers(X)
        
        # Transform using column transformer
        X_transformed = self.preprocessor.transform(X)
        
        # Handle date features
        if self.date_features:
            X_dates = self._extract_date_features(X)
            
            # Combine with transformed data
            if isinstance(X_transformed, np.ndarray):
                X_transformed = np.hstack([X_transformed, X_dates])
            else:
                # If sparse matrix, convert to dense first
                X_transformed = np.hstack([X_transformed.toarray(), X_dates])
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        # Return as DataFrame
        return pd.DataFrame(X_transformed, columns=feature_names)
    
    def _handle_outliers(self, X):
        """
        Handle outliers in numerical features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with outliers handled
        """
        X_copy = X.copy()
        
        for col in self.numerical_features:
            if col in X_copy.columns:
                # Calculate IQR
                Q1 = X_copy[col].quantile(0.25)
                Q3 = X_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                X_copy[col] = X_copy[col].clip(lower_bound, upper_bound)
        
        return X_copy
    
    def _extract_date_features(self, X):
        """
        Extract features from date columns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Extracted date features
        """
        date_features = []
        
        for col in self.date_features:
            if col in X.columns:
                # Convert to datetime
                dates = pd.to_datetime(X[col])
                
                # Extract features
                year = dates.dt.year
                month = dates.dt.month
                day = dates.dt.day
                day_of_week = dates.dt.dayofweek
                is_month_end = dates.dt.is_month_end.astype(int)
                is_month_start = dates.dt.is_month_start.astype(int)
                is_quarter_end = dates.dt.is_quarter_end.astype(int)
                
                # Stack features
                col_features = np.column_stack([
                    year, month, day, day_of_week, 
                    is_month_end, is_month_start, is_quarter_end
                ])
                
                date_features.append(col_features)
        
        # Combine all date features
        if date_features:
            return np.hstack(date_features)
        else:
            return np.empty((X.shape[0], 0))
    
    def _get_feature_names(self):
        """
        Get feature names after transformation.
        
        Returns:
        --------
        list
            List of feature names
        """
        feature_names = []
        
        # Get names from column transformer
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names.extend(self.preprocessor.get_feature_names_out())
        else:
            # Fallback for older scikit-learn versions
            for name, _, cols in self.preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    for col in cols:
                        feature_names.extend([f"{col}_{cat}" for cat in X[col].unique()])
        
        # Add date feature names
        for col in self.date_features:
            feature_names.extend([
                f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek",
                f"{col}_is_month_end", f"{col}_is_month_start", f"{col}_is_quarter_end"
            ])
        
        return feature_names


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts advanced features from market and options data.
    """
    def __init__(self, extract_technical=True, extract_volatility=True, 
                 extract_options=True, extract_market=True):
        """
        Initialize the feature extractor.
        
        Parameters:
        -----------
        extract_technical : bool
            Whether to extract technical indicator features
        extract_volatility : bool
            Whether to extract volatility regime features
        extract_options : bool
            Whether to extract options-specific features
        extract_market : bool
            Whether to extract market context features
        """
        self.extract_technical = extract_technical
        self.extract_volatility = extract_volatility
        self.extract_options = extract_options
        self.extract_market = extract_market
        
    def fit(self, X, y=None):
        """
        Fit the feature extractor to the data.
        
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
        # Nothing to fit for now
        return self
    
    def transform(self, X):
        """
        Transform the data by extracting features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with extracted features
        """
        X_copy = X.copy()
        
        # Extract features based on configuration
        if self.extract_technical:
            X_copy = self._extract_technical_features(X_copy)
            
        if self.extract_volatility:
            X_copy = self._extract_volatility_features(X_copy)
            
        if self.extract_options:
            X_copy = self._extract_options_features(X_copy)
            
        if self.extract_market:
            X_copy = self._extract_market_features(X_copy)
        
        return X_copy
    
    def _extract_technical_features(self, X):
        """
        Extract features from technical indicators.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with technical features
        """
        # Check if we have price data
        if not all(col in X.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return X
        
        # Calculate rate of change for existing indicators
        for col in X.columns:
            if col.startswith('rsi_') or col.startswith('macd_') or col.startswith('ema_'):
                X[f"{col}_roc"] = X[col].pct_change()
                
        # Calculate indicator divergence
        if 'rsi_14' in X.columns and 'close' in X.columns:
            # RSI divergence
            price_higher = (X['close'] > X['close'].shift(1))
            rsi_lower = (X['rsi_14'] < X['rsi_14'].shift(1))
            X['bearish_divergence'] = (price_higher & rsi_lower).astype(int)
            
            price_lower = (X['close'] < X['close'].shift(1))
            rsi_higher = (X['rsi_14'] > X['rsi_14'].shift(1))
            X['bullish_divergence'] = (price_lower & rsi_higher).astype(int)
        
        # Calculate indicator crossovers
        if 'ema_9' in X.columns and 'ema_21' in X.columns:
            X['ema_crossover'] = ((X['ema_9'] > X['ema_21']) & 
                                 (X['ema_9'].shift(1) <= X['ema_21'].shift(1))).astype(int)
            X['ema_crossunder'] = ((X['ema_9'] < X['ema_21']) & 
                                  (X['ema_9'].shift(1) >= X['ema_21'].shift(1))).astype(int)
        
        # Calculate price patterns
        if all(col in X.columns for col in ['open', 'high', 'low', 'close']):
            # Doji pattern
            body_size = abs(X['close'] - X['open'])
            total_range = X['high'] - X['low']
            X['doji'] = (body_size / total_range < 0.1).astype(int)
            
            # Hammer pattern
            lower_wick = np.minimum(X['open'], X['close']) - X['low']
            upper_wick = X['high'] - np.maximum(X['open'], X['close'])
            X['hammer'] = ((lower_wick > 2 * body_size) & 
                          (upper_wick < 0.2 * body_size) & 
                          (body_size / total_range < 0.3)).astype(int)
        
        return X
    
    def _extract_volatility_features(self, X):
        """
        Extract volatility regime features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with volatility features
        """
        # Check if we have price data
        if 'close' not in X.columns:
            return X
        
        # Calculate historical volatility
        if 'close' in X.columns:
            returns = X['close'].pct_change()
            X['hist_vol_5'] = returns.rolling(5).std() * np.sqrt(252)
            X['hist_vol_21'] = returns.rolling(21).std() * np.sqrt(252)
            
            # Volatility ratio
            X['vol_ratio'] = X['hist_vol_5'] / X['hist_vol_21']
            
            # Volatility regime
            X['high_vol_regime'] = (X['hist_vol_21'] > X['hist_vol_21'].rolling(63).mean()).astype(int)
            
            # Volatility trend
            X['vol_trend'] = (X['hist_vol_5'].diff() > 0).astype(int)
        
        # Calculate implied volatility features if available
        if 'impliedVolatility' in X.columns:
            X['iv_percentile'] = X['impliedVolatility'].rank(pct=True)
            
            if 'hist_vol_21' in X.columns:
                X['iv_hv_ratio'] = X['impliedVolatility'] / X['hist_vol_21']
        
        return X
    
    def _extract_options_features(self, X):
        """
        Extract options-specific features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with options features
        """
        # Check if we have options data
        if not all(col in X.columns for col in ['strikePrice', 'daysToExpiration', 'delta']):
            return X
        
        # Calculate moneyness
        if 'strikePrice' in X.columns and 'underlyingPrice' in X.columns:
            X['moneyness'] = X['strikePrice'] / X['underlyingPrice']
            
            # Moneyness categories
            X['deep_itm'] = (X['moneyness'] < 0.8).astype(int)
            X['itm'] = ((X['moneyness'] >= 0.8) & (X['moneyness'] < 0.95)).astype(int)
            X['atm'] = ((X['moneyness'] >= 0.95) & (X['moneyness'] <= 1.05)).astype(int)
            X['otm'] = ((X['moneyness'] > 1.05) & (X['moneyness'] <= 1.2)).astype(int)
            X['deep_otm'] = (X['moneyness'] > 1.2).astype(int)
        
        # Calculate time decay acceleration
        if 'theta' in X.columns and 'daysToExpiration' in X.columns:
            X['theta_per_day'] = X['theta'] / X['daysToExpiration']
            
        # Calculate risk-reward metrics
        if 'delta' in X.columns and 'gamma' in X.columns:
            X['gamma_exposure'] = X['gamma'] * X['delta']
            
        # Calculate liquidity metrics
        if 'volume' in X.columns and 'openInterest' in X.columns:
            X['volume_oi_ratio'] = X['volume'] / X['openInterest'].replace(0, 1)
            
        # Calculate expiration buckets
        if 'daysToExpiration' in X.columns:
            X['short_term'] = (X['daysToExpiration'] <= 30).astype(int)
            X['medium_term'] = ((X['daysToExpiration'] > 30) & 
                               (X['daysToExpiration'] <= 60)).astype(int)
            X['long_term'] = (X['daysToExpiration'] > 60).astype(int)
        
        return X
    
    def _extract_market_features(self, X):
        """
        Extract market context features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with market context features
        """
        # Check if we have market data
        if not any(col in X.columns for col in ['vix', 'market_trend', 'sector_trend']):
            return X
        
        # Calculate VIX-based features
        if 'vix' in X.columns:
            X['high_vix'] = (X['vix'] > 20).astype(int)
            X['extreme_vix'] = (X['vix'] > 30).astype(int)
            
            # VIX trend
            X['vix_trend'] = (X['vix'].diff() > 0).astype(int)
        
        # Calculate market trend features
        if 'market_trend' in X.columns:
            X['market_bullish'] = (X['market_trend'] == 'bullish').astype(int)
            X['market_bearish'] = (X['market_trend'] == 'bearish').astype(int)
        
        # Calculate sector trend features
        if 'sector_trend' in X.columns:
            X['sector_bullish'] = (X['sector_trend'] == 'bullish').astype(int)
            X['sector_bearish'] = (X['sector_trend'] == 'bearish').astype(int)
            
        # Calculate trend alignment
        if 'market_trend' in X.columns and 'sector_trend' in X.columns:
            X['trend_alignment'] = (X['market_trend'] == X['sector_trend']).astype(int)
        
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects the most relevant features for model training.
    """
    def __init__(self, method='k_best', k=20, threshold=0.05, use_pca=False, n_components=0.95):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        method : str
            Feature selection method ('k_best', 'mutual_info', 'correlation')
        k : int
            Number of top features to select
        threshold : float
            Threshold for correlation-based selection
        use_pca : bool
            Whether to apply PCA after feature selection
        n_components : float or int
            Number of components for PCA
        """
        self.method = method
        self.k = k
        self.threshold = threshold
        self.use_pca = use_pca
        self.n_components = n_components
        self.selector = None
        self.pca = None
        self.selected_features = None
        
    def fit(self, X, y=None):
        """
        Fit the feature selector to the data.
        
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
        # Handle timestamp columns by converting to numeric
        X_processed = self._preprocess_data(X)
        
        # Select features based on method
        if self.method == 'k_best':
            if y is not None:
                self.selector = SelectKBest(f_regression, k=self.k)
                self.selector.fit(X_processed, y)
            else:
                # If no target, use variance threshold
                from sklearn.feature_selection import VarianceThreshold
                self.selector = VarianceThreshold()
                self.selector.fit(X_processed)
                
        elif self.method == 'mutual_info':
            if y is not None:
                self.selector = SelectKBest(mutual_info_regression, k=self.k)
                self.selector.fit(X_processed, y)
            else:
                # If no target, use variance threshold
                from sklearn.feature_selection import VarianceThreshold
                self.selector = VarianceThreshold()
                self.selector.fit(X_processed)
                
        elif self.method == 'correlation':
            # Correlation-based selection doesn't need a selector object
            if y is not None:
                # Calculate correlation with target
                corr = pd.DataFrame(X_processed).corrwith(pd.Series(y)).abs()
                # Select features above threshold
                self.selected_features = corr[corr > self.threshold].index.tolist()
                # Limit to top k if specified
                if self.k is not None:
                    self.selected_features = corr.nlargest(self.k).index.tolist()
            else:
                # If no target, select based on variance
                from sklearn.feature_selection import VarianceThreshold
                self.selector = VarianceThreshold()
                self.selector.fit(X_processed)
        
        # Apply PCA if requested
        if self.use_pca:
            if self.method == 'correlation' and self.selected_features:
                # Apply PCA to selected features
                X_selected = X_processed[self.selected_features]
                self.pca = PCA(n_components=self.n_components)
                self.pca.fit(X_selected)
            else:
                # Apply PCA to all features or those selected by selector
                if self.selector:
                    X_selected = self.selector.transform(X_processed)
                else:
                    X_selected = X_processed
                self.pca = PCA(n_components=self.n_components)
                self.pca.fit(X_selected)
        
        return self
    
    def transform(self, X):
        """
        Transform the data by selecting features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Data with selected features
        """
        # Handle timestamp columns by converting to numeric
        X_processed = self._preprocess_data(X)
        
        # Select features based on method
        if self.method == 'correlation' and self.selected_features:
            X_selected = X_processed[self.selected_features]
        elif self.selector:
            X_selected = self.selector.transform(X_processed)
            
            # Convert to DataFrame with feature names if possible
            if hasattr(self.selector, 'get_support'):
                selected_cols = X_processed.columns[self.selector.get_support()]
                X_selected = pd.DataFrame(X_selected, columns=selected_cols)
        else:
            X_selected = X_processed
        
        # Apply PCA if requested
        if self.use_pca and self.pca:
            X_pca = self.pca.transform(X_selected)
            
            # Create feature names for PCA components
            pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            
            # Return as DataFrame
            return pd.DataFrame(X_pca, columns=pca_cols)
        
        return X_selected
    
    def _preprocess_data(self, X):
        """
        Preprocess data for feature selection, handling timestamps and non-numeric data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed data suitable for feature selection
        """
        X_processed = X.copy()
        
        # Convert DataFrame to numeric, handling timestamps and categorical data
        for col in X_processed.columns:
            # Check if column contains timestamps
            if pd.api.types.is_datetime64_any_dtype(X_processed[col]):
                # Convert timestamps to Unix time (seconds since epoch)
                X_processed[col] = X_processed[col].astype(np.int64) // 10**9
            elif not pd.api.types.is_numeric_dtype(X_processed[col]):
                # For non-numeric columns, try to convert to numeric or drop
                try:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                except:
                    # If conversion fails, drop the column
                    X_processed = X_processed.drop(columns=[col])
        
        # Fill any remaining NaN values
        X_processed = X_processed.fillna(0)
        
        return X_processed


class FeatureEngineeringPipeline:
    """
    Complete pipeline for feature engineering.
    Combines preprocessing, feature extraction, and feature selection.
    """
    def __init__(self, numerical_features=None, categorical_features=None, 
                 date_features=None, normalization='standard', handle_outliers=True,
                 extract_technical=True, extract_volatility=True, 
                 extract_options=True, extract_market=True,
                 selection_method='k_best', k=20, use_pca=False):
        """
        Initialize the feature engineering pipeline.
        
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
        extract_technical : bool
            Whether to extract technical indicator features
        extract_volatility : bool
            Whether to extract volatility regime features
        extract_options : bool
            Whether to extract options-specific features
        extract_market : bool
            Whether to extract market context features
        selection_method : str
            Feature selection method ('k_best', 'mutual_info', 'correlation')
        k : int
            Number of top features to select
        use_pca : bool
            Whether to apply PCA after feature selection
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.date_features = date_features
        self.normalization = normalization
        self.handle_outliers = handle_outliers
        self.extract_technical = extract_technical
        self.extract_volatility = extract_volatility
        self.extract_options = extract_options
        self.extract_market = extract_market
        self.selection_method = selection_method
        self.k = k
        self.use_pca = use_pca
        
        # Create pipeline components
        self.preprocessor = DataPreprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            date_features=date_features,
            normalization=normalization,
            handle_outliers=handle_outliers
        )
        
        self.feature_extractor = FeatureExtractor(
            extract_technical=extract_technical,
            extract_volatility=extract_volatility,
            extract_options=extract_options,
            extract_market=extract_market
        )
        
        self.feature_selector = FeatureSelector(
            method=selection_method,
            k=k,
            use_pca=use_pca
        )
        
        # Create scikit-learn pipeline
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('feature_extractor', self.feature_extractor),
            ('feature_selector', self.feature_selector)
        ])
        
    def fit(self, X, y=None):
        """
        Fit the pipeline to the data.
        
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
        Transform the data using the pipeline.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Transformed data
        """
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit the pipeline to the data and transform it.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        y : array-like, optional
            Target values
            
        Returns:
        --------
        pandas.DataFrame
            Transformed data
        """
        return self.pipeline.fit_transform(X, y)
    
    def get_feature_names(self):
        """
        Get the names of the features after transformation.
        
        Returns:
        --------
        list
            List of feature names
        """
        # Try to get feature names from the last step
        if hasattr(self.pipeline.steps[-1][1], 'get_feature_names_out'):
            return self.pipeline.steps[-1][1].get_feature_names_out()
        
        # If PCA was used, return PC names
        if self.use_pca:
            n_components = self.pipeline.steps[-1][1].pca.n_components_
            return [f'PC{i+1}' for i in range(n_components)]
        
        # Otherwise, try to get selected feature names
        if self.selection_method == 'correlation':
            return self.pipeline.steps[-1][1].selected_features
        
        # Fallback
        return None
