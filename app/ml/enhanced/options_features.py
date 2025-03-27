"""
Enhanced Options-Specific Feature Engineering.
Provides specialized transformers for extracting features from options data.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import datetime as dt

class OptionsFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts options-specific features from options chain data.
    Focuses on features relevant for options trading strategies.
    """
    def __init__(self, include_greeks=True, include_iv_features=True, 
                 include_term_structure=True, include_skew=True,
                 include_moneyness=True, include_time_features=True):
        """
        Initialize the options feature extractor.
        
        Parameters:
        -----------
        include_greeks : bool
            Whether to include Greek-derived features
        include_iv_features : bool
            Whether to include implied volatility features
        include_term_structure : bool
            Whether to include term structure features
        include_skew : bool
            Whether to include volatility skew features
        include_moneyness : bool
            Whether to include moneyness-based features
        include_time_features : bool
            Whether to include time-to-expiration features
        """
        self.include_greeks = include_greeks
        self.include_iv_features = include_iv_features
        self.include_term_structure = include_term_structure
        self.include_skew = include_skew
        self.include_moneyness = include_moneyness
        self.include_time_features = include_time_features
        
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Options data
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
        Transform the options data by extracting relevant features.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Options data with columns like strike, bid, ask, delta, gamma, etc.
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            Data with extracted options features
        """
        # Convert to DataFrame if it's a numpy array
        input_is_df = isinstance(X, pd.DataFrame)
        if not input_is_df:
            # Try to convert to DataFrame
            if hasattr(self, 'feature_names_in_'):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                # If we don't have feature names, we can't extract features properly
                return X
        
        # Make a copy to avoid modifying the original
        X_transformed = X.copy()
        
        # Extract features based on configuration
        if self.include_greeks:
            X_transformed = self._extract_greek_features(X_transformed)
        
        if self.include_iv_features:
            X_transformed = self._extract_iv_features(X_transformed)
        
        if self.include_term_structure:
            X_transformed = self._extract_term_structure(X_transformed)
        
        if self.include_skew:
            X_transformed = self._extract_skew_features(X_transformed)
        
        if self.include_moneyness:
            X_transformed = self._extract_moneyness_features(X_transformed)
        
        if self.include_time_features:
            X_transformed = self._extract_time_features(X_transformed)
        
        # Return in the same format as input
        if not input_is_df:
            return X_transformed.values
        
        return X_transformed
    
    def _extract_greek_features(self, X):
        """
        Extract features derived from options Greeks.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with Greek-derived features
        """
        # Ensure required columns exist
        required_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for col in required_cols:
            if col not in X.columns:
                # Skip this feature if data is missing
                return X
        
        # Calculate Greek ratios and derivatives
        X['gamma_to_delta_ratio'] = np.abs(X['gamma'] / (X['delta'] + 1e-10))
        X['theta_to_vega_ratio'] = np.abs(X['theta'] / (X['vega'] + 1e-10))
        X['normalized_gamma'] = X['gamma'] * X['underlyingPrice']**2 / 100
        X['normalized_vega'] = X['vega'] / (X['underlyingPrice'] * 0.01)
        X['delta_dollars'] = X['delta'] * X['underlyingPrice']
        X['gamma_dollars'] = X['gamma'] * X['underlyingPrice']**2 * 0.01
        X['theta_dollars'] = X['theta'] / 365  # Daily theta in dollars
        X['vega_dollars'] = X['vega'] * X['underlyingPrice'] * 0.01
        
        # Calculate risk-reward metrics
        X['gamma_per_theta'] = np.abs(X['gamma'] / (X['theta'] + 1e-10))
        X['vega_per_theta'] = np.abs(X['vega'] / (X['theta'] + 1e-10))
        
        return X
    
    def _extract_iv_features(self, X):
        """
        Extract features related to implied volatility.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with IV-related features
        """
        # Ensure required columns exist
        if 'impliedVolatility' not in X.columns:
            return X
        
        # Basic IV features
        X['iv_percentile'] = X['impliedVolatility'].rank(pct=True)
        
        # IV relative to historical volatility (if available)
        if 'historicalVolatility' in X.columns:
            X['iv_to_hv_ratio'] = X['impliedVolatility'] / (X['historicalVolatility'] + 1e-10)
            X['iv_hv_difference'] = X['impliedVolatility'] - X['historicalVolatility']
        
        # IV term structure (if multiple expirations)
        if 'daysToExpiration' in X.columns and len(X['daysToExpiration'].unique()) > 1:
            # Group by strike and calculate IV slope
            try:
                iv_term = X.groupby('strike')[['daysToExpiration', 'impliedVolatility']].apply(
                    lambda g: np.polyfit(g['daysToExpiration'], g['impliedVolatility'], 1)[0] 
                    if len(g) > 1 else np.nan
                ).rename('iv_term_slope')
                
                # Merge back to original dataframe
                X = X.merge(iv_term, left_on='strike', right_index=True, how='left')
            except Exception:
                # Skip if there's an error in the calculation
                pass
        
        return X
    
    def _extract_term_structure(self, X):
        """
        Extract features related to term structure.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with term structure features
        """
        # Ensure required columns exist
        if 'daysToExpiration' not in X.columns or 'impliedVolatility' not in X.columns:
            return X
        
        # Skip if not enough different expirations
        if len(X['daysToExpiration'].unique()) < 2:
            return X
        
        try:
            # Calculate term structure metrics
            # Group by option type and underlying price
            if 'putCall' in X.columns and 'underlyingPrice' in X.columns:
                groups = X.groupby(['putCall', 'underlyingPrice'])
                
                # Calculate term structure slope for each group
                term_slopes = groups.apply(lambda g: self._calculate_term_slope(g))
                
                # Merge back if we have results
                if not term_slopes.empty:
                    # Convert to DataFrame if it's a Series
                    if isinstance(term_slopes, pd.Series):
                        term_slopes = term_slopes.reset_index()
                        term_slopes.columns = list(term_slopes.columns[:-1]) + ['term_structure_slope']
                    
                    # Merge back to original DataFrame
                    X = X.merge(term_slopes, on=['putCall', 'underlyingPrice'], how='left')
        except Exception:
            # Skip if there's an error in the calculation
            pass
        
        return X
    
    def _calculate_term_slope(self, group):
        """
        Calculate the slope of the term structure for a group.
        
        Parameters:
        -----------
        group : pandas.DataFrame
            Group of options with same type and underlying price
            
        Returns:
        --------
        float
            Slope of the term structure
        """
        if len(group) < 2:
            return np.nan
        
        # Sort by days to expiration
        group = group.sort_values('daysToExpiration')
        
        # Calculate slope using linear regression
        days = group['daysToExpiration'].values
        ivs = group['impliedVolatility'].values
        
        # Use log of days for better fit
        log_days = np.log(days + 1)
        
        # Fit line
        slope, _ = np.polyfit(log_days, ivs, 1)
        
        return slope
    
    def _extract_skew_features(self, X):
        """
        Extract features related to volatility skew.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with skew features
        """
        # Ensure required columns exist
        if 'strike' not in X.columns or 'impliedVolatility' not in X.columns or 'underlyingPrice' not in X.columns:
            return X
        
        # Calculate moneyness
        X['moneyness'] = X['strike'] / X['underlyingPrice']
        
        try:
            # Group by expiration date and option type
            if 'expirationDate' in X.columns and 'putCall' in X.columns:
                groups = X.groupby(['expirationDate', 'putCall'])
                
                # Calculate skew for each group
                skew_data = groups.apply(lambda g: self._calculate_skew(g))
                
                # Merge back if we have results
                if not skew_data.empty:
                    # Convert to DataFrame if it's a Series
                    if isinstance(skew_data, pd.Series):
                        skew_data = skew_data.reset_index()
                        skew_data.columns = list(skew_data.columns[:-1]) + ['volatility_skew']
                    
                    # Merge back to original DataFrame
                    X = X.merge(skew_data, on=['expirationDate', 'putCall'], how='left')
        except Exception:
            # Skip if there's an error in the calculation
            pass
        
        return X
    
    def _calculate_skew(self, group):
        """
        Calculate the volatility skew for a group.
        
        Parameters:
        -----------
        group : pandas.DataFrame
            Group of options with same expiration and type
            
        Returns:
        --------
        float
            Volatility skew measure
        """
        if len(group) < 3:
            return np.nan
        
        # Sort by moneyness
        group = group.sort_values('moneyness')
        
        # Calculate skew using linear regression
        moneyness = group['moneyness'].values
        ivs = group['impliedVolatility'].values
        
        # Fit line
        slope, _ = np.polyfit(moneyness, ivs, 1)
        
        return slope
    
    def _extract_moneyness_features(self, X):
        """
        Extract features related to option moneyness.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with moneyness features
        """
        # Ensure required columns exist
        if 'strike' not in X.columns or 'underlyingPrice' not in X.columns:
            return X
        
        # Calculate moneyness if not already done
        if 'moneyness' not in X.columns:
            X['moneyness'] = X['strike'] / X['underlyingPrice']
        
        # Calculate moneyness categories
        X['moneyness_category'] = pd.cut(
            X['moneyness'],
            bins=[0, 0.8, 0.95, 1.05, 1.2, float('inf')],
            labels=['deep_itm', 'itm', 'atm', 'otm', 'deep_otm']
        )
        
        # Calculate normalized moneyness based on option type
        if 'putCall' in X.columns:
            X['normalized_moneyness'] = np.where(
                X['putCall'].str.upper() == 'CALL',
                X['moneyness'],
                2 - X['moneyness']  # For puts, transform to make ITM/OTM consistent with calls
            )
        
        # Calculate distance from ATM in standard deviations
        if 'impliedVolatility' in X.columns and 'daysToExpiration' in X.columns:
            # Calculate expected move based on IV
            X['expected_move_pct'] = X['impliedVolatility'] * np.sqrt(X['daysToExpiration'] / 365)
            X['strike_distance_in_sd'] = np.abs(X['moneyness'] - 1) / (X['expected_move_pct'] + 1e-10)
        
        return X
    
    def _extract_time_features(self, X):
        """
        Extract features related to time to expiration.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with time features
        """
        # Ensure required columns exist
        if 'daysToExpiration' not in X.columns:
            return X
        
        # Calculate time decay acceleration
        X['sqrt_time'] = np.sqrt(X['daysToExpiration'] / 365)
        X['log_time'] = np.log(X['daysToExpiration'] + 1)
        
        # Calculate time categories
        X['time_category'] = pd.cut(
            X['daysToExpiration'],
            bins=[-1, 7, 30, 90, 180, float('inf')],
            labels=['weekly', 'monthly', 'quarterly', 'half_year', 'leap']
        )
        
        # Calculate theta acceleration
        if 'theta' in X.columns:
            X['theta_per_day'] = X['theta'] / (X['daysToExpiration'] + 1e-10)
        
        # Calculate time value if price data available
        if all(col in X.columns for col in ['bid', 'ask', 'strike', 'underlyingPrice', 'putCall']):
            # Calculate mid price
            X['mid_price'] = (X['bid'] + X['ask']) / 2
            
            # Calculate intrinsic value
            X['intrinsic_value'] = np.where(
                X['putCall'].str.upper() == 'CALL',
                np.maximum(0, X['underlyingPrice'] - X['strike']),
                np.maximum(0, X['strike'] - X['underlyingPrice'])
            )
            
            # Calculate time value
            X['time_value'] = X['mid_price'] - X['intrinsic_value']
            X['time_value_pct'] = X['time_value'] / (X['mid_price'] + 1e-10)
        
        return X


class OptionSpreadFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features related to option spreads and pricing relationships.
    """
    def __init__(self, include_bid_ask=True, include_volume=True, include_open_interest=True):
        """
        Initialize the option spread feature extractor.
        
        Parameters:
        -----------
        include_bid_ask : bool
            Whether to include bid-ask spread features
        include_volume : bool
            Whether to include volume-related features
        include_open_interest : bool
            Whether to include open interest features
        """
        self.include_bid_ask = include_bid_ask
        self.include_volume = include_volume
        self.include_open_interest = include_open_interest
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Options data
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
        Transform the options data by extracting spread features.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Options data
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            Data with extracted spread features
        """
        # Convert to DataFrame if it's a numpy array
        input_is_df = isinstance(X, pd.DataFrame)
        if not input_is_df:
            # Try to convert to DataFrame
            if hasattr(self, 'feature_names_in_'):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                # If we don't have feature names, we can't extract features properly
                return X
        
        # Make a copy to avoid modifying the original
        X_transformed = X.copy()
        
        # Extract features based on configuration
        if self.include_bid_ask:
            X_transformed = self._extract_bid_ask_features(X_transformed)
        
        if self.include_volume:
            X_transformed = self._extract_volume_features(X_transformed)
        
        if self.include_open_interest:
            X_transformed = self._extract_open_interest_features(X_transformed)
        
        # Return in the same format as input
        if not input_is_df:
            return X_transformed.values
        
        return X_transformed
    
    def _extract_bid_ask_features(self, X):
        """
        Extract features related to bid-ask spreads.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with bid-ask features
        """
        # Ensure required columns exist
        if 'bid' not in X.columns or 'ask' not in X.columns:
            return X
        
        # Calculate basic bid-ask metrics
        X['bid_ask_spread'] = X['ask'] - X['bid']
        X['bid_ask_spread_pct'] = X['bid_ask_spread'] / ((X['bid'] + X['ask']) / 2 + 1e-10)
        X['mid_price'] = (X['bid'] + X['ask']) / 2
        
        # Calculate liquidity score (inverse of spread percentage)
        X['liquidity_score'] = 1 / (X['bid_ask_spread_pct'] + 1e-10)
        X['liquidity_score'] = X['liquidity_score'].clip(0, 100)  # Cap at 100
        
        return X
    
    def _extract_volume_features(self, X):
        """
        Extract features related to trading volume.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with volume features
        """
        # Ensure required columns exist
        if 'volume' not in X.columns:
            return X
        
        # Calculate volume metrics
        X['log_volume'] = np.log(X['volume'] + 1)
        X['volume_rank'] = X['volume'].rank(pct=True)
        
        # Calculate volume relative to open interest if available
        if 'openInterest' in X.columns:
            X['volume_to_oi_ratio'] = X['volume'] / (X['openInterest'] + 1e-10)
        
        # Group by expiration and calculate relative volume
        if 'expirationDate' in X.columns:
            try:
                # Calculate average volume by expiration
                avg_volume = X.groupby('expirationDate')['volume'].transform('mean')
                X['relative_volume'] = X['volume'] / (avg_volume + 1e-10)
            except Exception:
                # Skip if there's an error in the calculation
                pass
        
        return X
    
    def _extract_open_interest_features(self, X):
        """
        Extract features related to open interest.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with open interest features
        """
        # Ensure required columns exist
        if 'openInterest' not in X.columns:
            return X
        
        # Calculate open interest metrics
        X['log_open_interest'] = np.log(X['openInterest'] + 1)
        X['oi_rank'] = X['openInterest'].rank(pct=True)
        
        # Calculate put/call open interest ratio if possible
        if 'putCall' in X.columns:
            try:
                # Group by strike and expiration
                if 'strike' in X.columns and 'expirationDate' in X.columns:
                    # Calculate put and call OI for each strike/expiration
                    put_oi = X[X['putCall'].str.upper() == 'PUT'].set_index(['strike', 'expirationDate'])['openInterest']
                    call_oi = X[X['putCall'].str.upper() == 'CALL'].set_index(['strike', 'expirationDate'])['openInterest']
                    
                    # Create DataFrame with put/call OI
                    pc_ratio = pd.DataFrame({
                        'put_oi': put_oi,
                        'call_oi': call_oi
                    }).reset_index()
                    
                    # Calculate ratio
                    pc_ratio['pc_oi_ratio'] = pc_ratio['put_oi'] / (pc_ratio['call_oi'] + 1e-10)
                    
                    # Merge back to original DataFrame
                    X = X.merge(
                        pc_ratio[['strike', 'expirationDate', 'pc_oi_ratio']], 
                        on=['strike', 'expirationDate'], 
                        how='left'
                    )
            except Exception:
                # Skip if there's an error in the calculation
                pass
        
        return X


class MarketContextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features related to market context and conditions.
    """
    def __init__(self, include_market_indicators=True, include_sector_data=True, 
                 include_earnings=True, include_dividends=True):
        """
        Initialize the market context feature extractor.
        
        Parameters:
        -----------
        include_market_indicators : bool
            Whether to include market indicator features
        include_sector_data : bool
            Whether to include sector-related features
        include_earnings : bool
            Whether to include earnings-related features
        include_dividends : bool
            Whether to include dividend-related features
        """
        self.include_market_indicators = include_market_indicators
        self.include_sector_data = include_sector_data
        self.include_earnings = include_earnings
        self.include_dividends = include_dividends
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Options data
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
        Transform the options data by extracting market context features.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Options data
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            Data with extracted market context features
        """
        # Convert to DataFrame if it's a numpy array
        input_is_df = isinstance(X, pd.DataFrame)
        if not input_is_df:
            # Try to convert to DataFrame
            if hasattr(self, 'feature_names_in_'):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                # If we don't have feature names, we can't extract features properly
                return X
        
        # Make a copy to avoid modifying the original
        X_transformed = X.copy()
        
        # Extract features based on configuration
        if self.include_market_indicators:
            X_transformed = self._extract_market_indicators(X_transformed)
        
        if self.include_sector_data:
            X_transformed = self._extract_sector_data(X_transformed)
        
        if self.include_earnings:
            X_transformed = self._extract_earnings_features(X_transformed)
        
        if self.include_dividends:
            X_transformed = self._extract_dividend_features(X_transformed)
        
        # Return in the same format as input
        if not input_is_df:
            return X_transformed.values
        
        return X_transformed
    
    def _extract_market_indicators(self, X):
        """
        Extract features related to market indicators.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with market indicator features
        """
        # Check if market indicator columns are already present
        market_indicators = [
            'vix', 'market_iv', 'market_trend', 'market_breadth',
            'market_momentum', 'market_volatility'
        ]
        
        # If any indicators are present, use them
        if any(indicator in X.columns for indicator in market_indicators):
            # Calculate derived features from available indicators
            if 'vix' in X.columns:
                X['vix_percentile'] = X['vix'].rank(pct=True)
                
                # VIX relative to option IV if available
                if 'impliedVolatility' in X.columns:
                    X['iv_to_vix_ratio'] = X['impliedVolatility'] / (X['vix'] / 100 + 1e-10)
            
            if 'market_trend' in X.columns and 'market_volatility' in X.columns:
                # Calculate risk-adjusted trend
                X['risk_adjusted_trend'] = X['market_trend'] / (X['market_volatility'] + 1e-10)
        
        return X
    
    def _extract_sector_data(self, X):
        """
        Extract features related to sector performance.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sector features
        """
        # Check if sector data columns are already present
        sector_columns = [
            'sector', 'industry', 'sector_performance', 'industry_performance',
            'sector_volatility', 'industry_volatility', 'sector_correlation'
        ]
        
        # If any sector data is present, use it
        if any(col in X.columns for col in sector_columns):
            # Calculate derived features from available sector data
            if 'sector_performance' in X.columns and 'market_performance' in X.columns:
                X['sector_relative_performance'] = X['sector_performance'] - X['market_performance']
            
            if 'sector_volatility' in X.columns and 'market_volatility' in X.columns:
                X['sector_relative_volatility'] = X['sector_volatility'] / (X['market_volatility'] + 1e-10)
            
            if 'sector_correlation' in X.columns:
                X['sector_diversification_score'] = 1 - X['sector_correlation'].abs()
        
        return X
    
    def _extract_earnings_features(self, X):
        """
        Extract features related to earnings events.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with earnings features
        """
        # Check if earnings-related columns are already present
        earnings_columns = [
            'days_to_earnings', 'earnings_date', 'last_earnings_surprise',
            'earnings_estimate', 'earnings_whisper'
        ]
        
        # If any earnings data is present, use it
        if any(col in X.columns for col in earnings_columns):
            # Calculate derived features from available earnings data
            if 'days_to_earnings' in X.columns:
                # Create earnings proximity indicator
                X['earnings_proximity'] = np.exp(-X['days_to_earnings'] / 30)  # Exponential decay
                
                # Check if option expires after earnings
                if 'daysToExpiration' in X.columns:
                    X['expires_after_earnings'] = X['daysToExpiration'] > X['days_to_earnings']
            
            if 'last_earnings_surprise' in X.columns and 'impliedVolatility' in X.columns:
                # Calculate IV premium relative to earnings surprise
                X['iv_earnings_premium'] = X['impliedVolatility'] * np.abs(X['last_earnings_surprise'])
        
        return X
    
    def _extract_dividend_features(self, X):
        """
        Extract features related to dividends.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Options data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with dividend features
        """
        # Check if dividend-related columns are already present
        dividend_columns = [
            'dividend_yield', 'days_to_ex_dividend', 'ex_dividend_date',
            'dividend_amount', 'dividend_growth'
        ]
        
        # If any dividend data is present, use it
        if any(col in X.columns for col in dividend_columns):
            # Calculate derived features from available dividend data
            if 'days_to_ex_dividend' in X.columns:
                # Create dividend proximity indicator
                X['dividend_proximity'] = np.exp(-X['days_to_ex_dividend'] / 30)  # Exponential decay
                
                # Check if option expires after ex-dividend
                if 'daysToExpiration' in X.columns:
                    X['expires_after_ex_dividend'] = X['daysToExpiration'] > X['days_to_ex_dividend']
            
            if 'dividend_yield' in X.columns and 'impliedVolatility' in X.columns:
                # Calculate dividend-adjusted IV
                X['dividend_adjusted_iv'] = X['impliedVolatility'] - (X['dividend_yield'] / 100)
            
            if 'dividend_amount' in X.columns and 'underlyingPrice' in X.columns:
                # Calculate dividend as percentage of price
                X['dividend_pct'] = (X['dividend_amount'] / X['underlyingPrice']) * 100
        
        return X
