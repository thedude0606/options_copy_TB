"""
Enhanced Feature Extraction Module

This module provides feature extraction for options and underlying asset data.
It calculates technical indicators, volatility surface features, and term structure features.
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import norm

logger = logging.getLogger('feature_extraction')

class EnhancedFeatureExtractor:
    def __init__(self, db):
        self.db = db
        logger.info("Initialized enhanced feature extractor")
        
    def extract_features(self, symbol, lookback_days=30):
        """Extract enhanced features for ML models"""
        logger.info(f"Extracting features for {symbol} with {lookback_days} days lookback")
        
        # Get underlying historical data
        underlying_data = self.db.get_historical_underlying(symbol, lookback_days)
        
        if underlying_data.empty:
            logger.warning(f"No historical data available for {symbol}")
            return {}
            
        # Get current options data
        options_data = self.db.get_latest_options(symbol)
        
        # Calculate technical indicators
        technical_features = self._calculate_technical_indicators(underlying_data)
        
        # If options data is available, calculate options-specific features
        if not options_data.empty:
            # Calculate volatility surface features
            vol_surface_features = self._calculate_volatility_surface(options_data)
            
            # Calculate term structure features
            term_structure_features = self._calculate_term_structure(options_data)
            
            # Combine all features
            all_features = {
                **technical_features,
                **vol_surface_features,
                **term_structure_features
            }
        else:
            logger.warning(f"No options data available for {symbol}, using only technical features")
            all_features = technical_features
        
        logger.info(f"Extracted {len(all_features)} features for {symbol}")
        return all_features
        
    def _calculate_technical_indicators(self, data):
        """Calculate technical indicators from price data"""
        features = {}
        
        try:
            # Ensure we have a copy to avoid modifying the original
            df = data.copy()
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Volatility (20-day)
            features['historical_volatility'] = df['returns'].rolling(20).std().iloc[-1] * np.sqrt(252) if len(df) >= 20 else np.nan
            
            # RSI (14-day)
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = rsi.iloc[-1]
            else:
                features['rsi'] = np.nan
            
            # Moving Averages
            features['sma_20'] = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else np.nan
            features['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else np.nan
            features['sma_200'] = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else np.nan
            
            # MACD
            if len(df) >= 26:
                features['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
                features['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]
                features['macd'] = features['ema_12'] - features['ema_26']
                features['macd_signal'] = pd.Series(features['macd']).ewm(span=9).mean().iloc[-1] if 'macd' in features else np.nan
            else:
                features['macd'] = np.nan
                features['macd_signal'] = np.nan
            
            # Bollinger Bands
            if len(df) >= 20:
                features['bollinger_mid'] = df['close'].rolling(20).mean().iloc[-1]
                features['bollinger_std'] = df['close'].rolling(20).std().iloc[-1]
                features['bollinger_upper'] = features['bollinger_mid'] + 2 * features['bollinger_std']
                features['bollinger_lower'] = features['bollinger_mid'] - 2 * features['bollinger_std']
                
                # Current price position within Bollinger Bands
                current_price = df['close'].iloc[-1]
                band_width = features['bollinger_upper'] - features['bollinger_lower']
                if band_width > 0:
                    features['bb_position'] = (current_price - features['bollinger_lower']) / band_width
                else:
                    features['bb_position'] = 0.5
            else:
                features['bollinger_mid'] = np.nan
                features['bollinger_std'] = np.nan
                features['bollinger_upper'] = np.nan
                features['bollinger_lower'] = np.nan
                features['bb_position'] = np.nan
                
            # Trend features
            if len(df) >= 10:
                # Price momentum (10-day)
                features['momentum_10d'] = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100
                
                # Directional movement
                features['price_direction'] = 1 if df['close'].iloc[-1] > df['close'].iloc[-2] else -1
                
                # Trend strength
                if 'sma_20' in features and 'sma_50' in features and not np.isnan(features['sma_20']) and not np.isnan(features['sma_50']):
                    features['trend_strength'] = features['sma_20'] / features['sma_50'] - 1
                else:
                    features['trend_strength'] = np.nan
            else:
                features['momentum_10d'] = np.nan
                features['price_direction'] = 0
                features['trend_strength'] = np.nan
                
            logger.info(f"Calculated {len(features)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return features
        
    def _calculate_volatility_surface(self, options_data):
        """Calculate volatility surface features from options data"""
        features = {}
        
        try:
            # Ensure we have data
            if options_data.empty:
                return features
                
            # Group by expiration and strike
            options_data['days_to_expiry'] = pd.to_datetime(options_data['expiration_date']) - pd.to_datetime(options_data['timestamp'])
            options_data['days_to_expiry'] = options_data['days_to_expiry'].dt.days
            
            # Get the underlying price
            if 'underlyingPrice' in options_data.columns:
                underlying_price = options_data['underlyingPrice'].iloc[0]
            else:
                # Estimate ATM strike as the middle of the range
                underlying_price = options_data['strike'].median()
            
            # Calculate moneyness
            options_data['moneyness'] = options_data['strike'] / underlying_price
            
            # Filter to reasonable moneyness range
            filtered_data = options_data[(options_data['moneyness'] >= 0.8) & (options_data['moneyness'] <= 1.2)]
            
            if filtered_data.empty:
                logger.warning("No options data in reasonable moneyness range")
                return features
                
            # Group by expiration and option type
            grouped = filtered_data.groupby(['days_to_expiry', 'option_type'])
            
            # Calculate IV skew for each expiration
            iv_skew = {}
            for (days, option_type), group in grouped:
                # Sort by moneyness
                sorted_group = group.sort_values('moneyness')
                
                # Find ATM option (closest to 1.0 moneyness)
                atm_idx = (sorted_group['moneyness'] - 1.0).abs().idxmin()
                atm_iv = sorted_group.loc[atm_idx, 'implied_volatility']
                
                # Find OTM options
                if option_type == 'PUT':
                    otm_options = sorted_group[sorted_group['moneyness'] < 0.95]
                else:
                    otm_options = sorted_group[sorted_group['moneyness'] > 1.05]
                    
                if not otm_options.empty:
                    # Calculate average OTM IV
                    otm_iv = otm_options['implied_volatility'].mean()
                    
                    # Calculate skew
                    iv_skew[(days, option_type)] = otm_iv - atm_iv
            
            # Calculate average skew for puts and calls
            put_skews = [skew for (days, opt_type), skew in iv_skew.items() if opt_type == 'PUT']
            call_skews = [skew for (days, opt_type), skew in iv_skew.items() if opt_type == 'CALL']
            
            if put_skews:
                features['put_iv_skew'] = np.mean(put_skews)
            if call_skews:
                features['call_iv_skew'] = np.mean(call_skews)
                
            # Calculate overall skew
            if put_skews and call_skews:
                features['iv_skew'] = np.mean(put_skews + call_skews)
                
            # Calculate IV by moneyness buckets
            moneyness_buckets = [(0.8, 0.9), (0.9, 0.95), (0.95, 1.0), (1.0, 1.05), (1.05, 1.1), (1.1, 1.2)]
            
            for lower, upper in moneyness_buckets:
                bucket_options = filtered_data[(filtered_data['moneyness'] >= lower) & (filtered_data['moneyness'] < upper)]
                if not bucket_options.empty:
                    features[f'iv_moneyness_{lower}_{upper}'] = bucket_options['implied_volatility'].mean()
            
            logger.info(f"Calculated {len(features)} volatility surface features")
            
        except Exception as e:
            logger.error(f"Error calculating volatility surface features: {e}")
            
        return features
        
    def _calculate_term_structure(self, options_data):
        """Calculate term structure features from options data"""
        features = {}
        
        try:
            # Ensure we have data
            if options_data.empty:
                return features
                
            # Calculate days to expiry if not already done
            if 'days_to_expiry' not in options_data.columns:
                options_data['days_to_expiry'] = pd.to_datetime(options_data['expiration_date']) - pd.to_datetime(options_data['timestamp'])
                options_data['days_to_expiry'] = options_data['days_to_expiry'].dt.days
            
            # Group by expiration
            grouped = options_data.groupby('days_to_expiry')
            
            # Calculate average IV by expiration
            iv_by_expiration = grouped['implied_volatility'].mean()
            
            if len(iv_by_expiration) > 1:
                # Sort by days to expiry
                iv_by_expiration = iv_by_expiration.sort_index()
                
                # Short-term IV (nearest expiration)
                features['short_term_iv'] = iv_by_expiration.iloc[0]
                
                # Mid-term IV (middle expiration if available)
                if len(iv_by_expiration) >= 3:
                    mid_idx = len(iv_by_expiration) // 2
                    features['mid_term_iv'] = iv_by_expiration.iloc[mid_idx]
                
                # Long-term IV (furthest expiration)
                features['long_term_iv'] = iv_by_expiration.iloc[-1]
                
                # Term structure slopes
                if 'mid_term_iv' in features:
                    features['short_mid_slope'] = (features['mid_term_iv'] - features['short_term_iv']) / (iv_by_expiration.index[len(iv_by_expiration) // 2] - iv_by_expiration.index[0])
                    features['mid_long_slope'] = (features['long_term_iv'] - features['mid_term_iv']) / (iv_by_expiration.index[-1] - iv_by_expiration.index[len(iv_by_expiration) // 2])
                else:
                    features['term_slope'] = (features['long_term_iv'] - features['short_term_iv']) / (iv_by_expiration.index[-1] - iv_by_expiration.index[0])
                
                # Overall term structure shape
                if len(iv_by_expiration) >= 3:
                    # Fit a quadratic curve to the term structure
                    x = iv_by_expiration.index.values
                    y = iv_by_expiration.values
                    coeffs = np.polyfit(x, y, 2)
                    
                    # Store coefficients
                    features['term_structure_a'] = coeffs[0]  # Quadratic term
                    features['term_structure_b'] = coeffs[1]  # Linear term
                    features['term_structure_c'] = coeffs[2]  # Constant term
                    
                    # Characterize the shape
                    features['term_structure_shape'] = 1 if coeffs[0] > 0 else -1  # 1 for convex, -1 for concave
            
            logger.info(f"Calculated {len(features)} term structure features")
            
        except Exception as e:
            logger.error(f"Error calculating term structure features: {e}")
            
        return features
