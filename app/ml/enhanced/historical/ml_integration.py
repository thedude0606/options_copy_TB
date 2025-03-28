"""
Enhanced ML Integration Module

This module provides ML integration for options trading recommendations.
It uses historical data, enhanced features, and synthetic data when needed.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger('enhanced_ml_integration')

class EnhancedMLIntegration:
    def __init__(self, config=None):
        """Initialize the enhanced ML integration with the given configuration."""
        self.config = config or {}
        logger.info("Using default configuration" if not config else "Using custom configuration")
        
        # Import our modules (import here to avoid circular imports)
        from options_db import OptionsDatabase
        from feature_extraction import EnhancedFeatureExtractor
        from synthetic_data import SyntheticOptionsGenerator
        
        # Initialize database
        self.db = OptionsDatabase()
        
        # Initialize feature extractor
        self.feature_extractor = EnhancedFeatureExtractor(self.db)
        
        # Initialize synthetic data generator
        self.synthetic_generator = SyntheticOptionsGenerator(self.db)
        
        # Initialize trading system components
        self._initialize_trading_system()
        
        logger.info("Enhanced trading system initialized successfully")
        logger.info("Enhanced ML Integration initialized")
        
    def _initialize_trading_system(self):
        """Initialize the trading system components."""
        # Set up model parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.risk_tolerance = self.config.get('risk_tolerance', 'moderate')
        self.position_sizing_method = self.config.get('position_sizing', 'fixed_risk')
        
        # Define risk parameters based on risk tolerance
        risk_params = {
            'conservative': {'max_position_size': 0.02, 'stop_loss_pct': 0.5, 'take_profit_ratio': 1.5},
            'moderate': {'max_position_size': 0.05, 'stop_loss_pct': 0.7, 'take_profit_ratio': 2.0},
            'aggressive': {'max_position_size': 0.1, 'stop_loss_pct': 1.0, 'take_profit_ratio': 3.0}
        }
        
        self.risk_params = risk_params.get(self.risk_tolerance, risk_params['moderate'])
        
    def predict(self, data):
        """Generate ML predictions based on enhanced features.
        
        Args:
            data (dict): Input data containing symbol and other parameters
            
        Returns:
            dict: Prediction results with confidence scores
        """
        logger.info("Generating ML predictions")
        
        try:
            # Extract symbol from data
            if isinstance(data, dict):
                symbol = data.get('symbol')
                if not symbol and 'symbols' in data and data['symbols']:
                    symbol = data['symbols'][0]
            elif hasattr(data, 'symbol'):
                symbol = data.symbol
            else:
                logger.warning("No symbol provided for prediction")
                return {'confidence': 0, 'prediction': 'neutral', 'details': {}}
                
            # Get historical options data if available
            historical_options = self.db.get_historical_options(symbol)
            
            # If no real historical data, use synthetic data
            if historical_options.empty:
                logger.info(f"No historical options data for {symbol}, generating synthetic data")
                # Get last 60 days of underlying data
                end_date = datetime.now().isoformat()
                start_date = (datetime.now() - timedelta(days=60)).isoformat()
                historical_options = self.synthetic_generator.generate_synthetic_history(
                    symbol, start_date=start_date, end_date=end_date
                )
            
            # Extract enhanced features
            features = self.feature_extractor.extract_features(symbol, lookback_days=30)
            
            # If we have options data, add it to the features
            if 'options_data' in data:
                options_data = data['options_data']
                if not isinstance(options_data, pd.DataFrame):
                    logger.warning("Options data provided is not a DataFrame")
                else:
                    logger.info(f"Using provided options data with {len(options_data)} records")
                    # Add volatility surface features
                    vol_features = self.feature_extractor._calculate_volatility_surface(options_data)
                    features.update(vol_features)
                    
                    # Add term structure features
                    term_features = self.feature_extractor._calculate_term_structure(options_data)
                    features.update(term_features)
            else:
                logger.warning("No options data provided for prediction")
            
            # Prepare market context features
            if 'market_data' in data and isinstance(data['market_data'], dict):
                market_data = data['market_data']
                for key, value in market_data.items():
                    features[f'market_{key}'] = value
            
            # Generate prediction based on features
            prediction_result = self._generate_prediction(features)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return {'confidence': 0, 'prediction': 'neutral', 'details': {}}
    
    def _generate_prediction(self, features):
        """Generate a prediction based on the extracted features.
        
        Args:
            features (dict): Extracted features
            
        Returns:
            dict: Prediction results
        """
        # This is a simplified prediction model
        # In a real implementation, you would use a trained ML model
        
        prediction = {'confidence': 0, 'prediction': 'neutral', 'details': {}}
        
        try:
            # Calculate directional signals
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI signals
            if 'rsi' in features and not np.isnan(features['rsi']):
                rsi = features['rsi']
                if rsi < 30:
                    bullish_signals += 1
                    prediction['details']['rsi'] = 'oversold'
                elif rsi > 70:
                    bearish_signals += 1
                    prediction['details']['rsi'] = 'overbought'
                else:
                    prediction['details']['rsi'] = 'neutral'
            
            # Moving average signals
            if 'sma_20' in features and 'sma_50' in features and not np.isnan(features['sma_20']) and not np.isnan(features['sma_50']):
                if features['sma_20'] > features['sma_50']:
                    bullish_signals += 1
                    prediction['details']['ma_crossover'] = 'bullish'
                else:
                    bearish_signals += 1
                    prediction['details']['ma_crossover'] = 'bearish'
            
            # MACD signals
            if 'macd' in features and 'macd_signal' in features and not np.isnan(features['macd']) and not np.isnan(features['macd_signal']):
                if features['macd'] > features['macd_signal']:
                    bullish_signals += 1
                    prediction['details']['macd'] = 'bullish'
                else:
                    bearish_signals += 1
                    prediction['details']['macd'] = 'bearish'
            
            # Bollinger Band signals
            if all(k in features for k in ['bollinger_upper', 'bollinger_lower', 'bollinger_mid']) and 'bb_position' in features:
                bb_pos = features['bb_position']
                if not np.isnan(bb_pos):
                    if bb_pos < 0.2:
                        bullish_signals += 1
                        prediction['details']['bollinger'] = 'oversold'
                    elif bb_pos > 0.8:
                        bearish_signals += 1
                        prediction['details']['bollinger'] = 'overbought'
                    else:
                        prediction['details']['bollinger'] = 'neutral'
            
            # Volatility skew signals (if available)
            if 'put_iv_skew' in features and not np.isnan(features['put_iv_skew']):
                if features['put_iv_skew'] > 0.05:  # High put skew indicates fear
                    bearish_signals += 1
                    prediction['details']['iv_skew'] = 'bearish'
                elif features['put_iv_skew'] < -0.02:  # Negative put skew is unusual and can be bullish
                    bullish_signals += 1
                    prediction['details']['iv_skew'] = 'bullish'
            
            # Term structure signals (if available)
            if 'term_slope' in features and not np.isnan(features['term_slope']):
                if features['term_slope'] > 0.002:  # Steep upward slope
                    bullish_signals += 1
                    prediction['details']['term_structure'] = 'bullish'
                elif features['term_slope'] < -0.001:  # Inverted term structure
                    bearish_signals += 1
                    prediction['details']['term_structure'] = 'bearish'
            
            # Calculate overall direction and confidence
            total_signals = bullish_signals + bearish_signals
            if total_signals > 0:
                if bullish_signals > bearish_signals:
                    prediction['prediction'] = 'bullish'
                    prediction['confidence'] = min(0.5 + (bullish_signals - bearish_signals) / (2 * total_signals), 0.95)
                elif bearish_signals > bullish_signals:
                    prediction['prediction'] = 'bearish'
                    prediction['confidence'] = min(0.5 + (bearish_signals - bullish_signals) / (2 * total_signals), 0.95)
                else:
                    prediction['prediction'] = 'neutral'
                    prediction['confidence'] = 0.5
            else:
                prediction['prediction'] = 'neutral'
                prediction['confidence'] = 0.5
            
            # Add feature values to details
            prediction['features'] = {k: v for k, v in features.items() if not np.isnan(v) if isinstance(v, (int, float))}
            
            logger.info(f"Generated prediction: {prediction['prediction']} with confidence {prediction['confidence']:.2f}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction generation: {e}")
            return {'confidence': 0, 'prediction': 'neutral', 'details': {}}
    
    def process_recommendation(self, recommendation):
        """Enhance a recommendation with risk management details.
        
        Args:
            recommendation (dict): The base recommendation
            
        Returns:
            dict: Enhanced recommendation with risk management details
        """
        try:
            # Ensure we have a valid recommendation
            if not recommendation or not isinstance(recommendation, dict):
                logger.warning("Invalid recommendation provided")
                return recommendation
                
            # Extract key values
            symbol = recommendation.get('symbol', '')
            option_type = recommendation.get('optionType', '')
            strike = recommendation.get('strikePrice', 0)
            entry_price = recommendation.get('entryPrice', 0)
            underlying_price = recommendation.get('underlyingPrice', 0)
            
            if not all([symbol, option_type, strike, entry_price, underlying_price]):
                logger.warning("Missing key values in recommendation")
                return recommendation
            
            # Calculate position sizing
            account_size = self.config.get('account_size', 100000)
            max_position_value = account_size * self.risk_params['max_position_size']
            
            # For options, we calculate contracts based on entry price
            if entry_price > 0:
                max_contracts = int(max_position_value / (entry_price * 100))  # Each contract is for 100 shares
            else:
                max_contracts = 0
                
            # Calculate stop loss and take profit levels
            if option_type.upper() == 'CALL':
                # For calls, stop loss is based on underlying price
                stop_loss_pct = self.risk_params['stop_loss_pct']
                stop_loss_price = max(0.01, entry_price * (1 - stop_loss_pct))
                
                # Take profit based on risk-reward ratio
                take_profit_ratio = self.risk_params['take_profit_ratio']
                take_profit_price = entry_price * (1 + (stop_loss_pct * take_profit_ratio))
                
                # Calculate underlying price levels
                stop_underlying = underlying_price * (1 - (stop_loss_pct / 2))
                target_underlying = underlying_price * (1 + (stop_loss_pct * take_profit_ratio / 2))
                
            else:  # PUT
                # For puts, stop loss is based on underlying price
                stop_loss_pct = self.risk_params['stop_loss_pct']
                stop_loss_price = max(0.01, entry_price * (1 - stop_loss_pct))
                
                # Take profit based on risk-reward ratio
                take_profit_ratio = self.risk_params['take_profit_ratio']
                take_profit_price = entry_price * (1 + (stop_loss_pct * take_profit_ratio))
                
                # Calculate underlying price levels
                stop_underlying = underlying_price * (1 + (stop_loss_pct / 2))
                target_underlying = underlying_price * (1 - (stop_loss_pct * take_profit_ratio / 2))
            
            # Calculate expected value
            win_probability = recommendation.get('probabilityOfProfit', 0.5)
            potential_gain = (take_profit_price - entry_price) * max_contracts * 100
            potential_loss = (entry_price - stop_loss_price) * max_contracts * 100
            expected_value = (win_probability * potential_gain) - ((1 - win_probability) * potential_loss)
            
            # Add risk management details to recommendation
            recommendation['riskManagement'] = {
                'positionSize': max_contracts,
                'maxPositionValue': max_position_value,
                'stopLossPrice': stop_loss_price,
                'takeProfitPrice': take_profit_price,
                'stopUnderlyingPrice': stop_underlying,
                'targetUnderlyingPrice': target_underlying,
                'expectedValue': expected_value,
                'riskRewardRatio': take_profit_ratio
            }
            
            logger.info(f"Added risk management details to recommendation for {symbol}")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error processing recommendation: {e}")
            return recommendation
