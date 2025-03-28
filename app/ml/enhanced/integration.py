"""
Integration script for enhanced ML and risk management components.
Connects the enhanced features with the existing dashboard.
"""
import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import json
import logging

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.ml.enhanced.trading_system import EnhancedTradingSystem
from app.data.options_symbol_parser import OptionsSymbolParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ml_integration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('enhanced_ml_integration')

class EnhancedMLIntegration:
    """
    Integration class for connecting enhanced ML and risk management features
    with the existing options trading dashboard.
    """
    def __init__(self, config_path=None):
        """
        Initialize the integration module.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.config_path = config_path
        self.trading_system = None
        self.options_parser = OptionsSymbolParser()
        self.initialize_system()
        logger.info("Enhanced ML Integration initialized")
    
    def initialize_system(self):
        """
        Initialize the enhanced trading system.
        """
        try:
            # Load configuration if provided
            if self.config_path and os.path.exists(self.config_path):
                logger.info(f"Loading configuration from {self.config_path}")
                self.trading_system = EnhancedTradingSystem()
                self.trading_system.load_configuration(self.config_path)
            else:
                logger.info("Using default configuration")
                self.trading_system = EnhancedTradingSystem()
            
            logger.info("Enhanced trading system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            raise
    
    def process_options_data(self, options_data, target=None):
        """
        Process options data through the enhanced feature engineering pipeline.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data to process
        target : array-like, optional
            Target values for supervised feature selection
            
        Returns:
        --------
        pandas.DataFrame
            Processed data with engineered features
        """
        try:
            logger.info(f"Processing options data with shape {options_data.shape}")
            processed_data = self.trading_system.process_data(options_data, target)
            logger.info(f"Processed data shape: {processed_data.shape}")
            return processed_data
        except Exception as e:
            logger.error(f"Error processing options data: {str(e)}")
            raise
    
    def train_prediction_models(self, options_data, target_column):
        """
        Train prediction models on options data.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data for training
        target_column : str
            Name of the target column
            
        Returns:
        --------
        dict
            Dictionary with trained model details
        """
        try:
            logger.info(f"Training prediction models on {len(options_data)} samples")
            
            # Extract features and target
            X = options_data.drop(target_column, axis=1)
            y = options_data[target_column]
            
            # Train models
            trained_models = self.trading_system.train_models(X, y)
            
            logger.info(f"Trained {len(trained_models)} models successfully")
            return trained_models
        except Exception as e:
            logger.error(f"Error training prediction models: {str(e)}")
            raise
    
    def generate_predictions(self, options_data):
        """
        Generate predictions for options data.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data to predict
            
        Returns:
        --------
        dict
            Dictionary with predictions and details
        """
        try:
            logger.info(f"Generating predictions for {len(options_data)} options")
            prediction_result = self.trading_system.predict(options_data)
            logger.info(f"Generated predictions using model: {prediction_result['model_details']['selected_model']}")
            return prediction_result
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def predict(self, features):
        """
        Generate ML predictions for the given features.
        This method is called by the EnhancedRecommendationEngine.
        
        Parameters:
        -----------
        features : dict
            Dictionary containing feature data including:
            - options_data: DataFrame with options data
            - technical_indicators: Dict of technical indicators
            - market_data: Dict of market context data
            - multi_timeframe_data: Dict of data from multiple timeframes
            
        Returns:
        --------
        dict
            Prediction results including:
            - prediction: Predicted return value
            - confidence: Confidence score (0-1)
            - risk_score: Risk assessment score (0-1)
            - probability_of_profit: Estimated probability of profit
        """
        try:
            logger.info("Generating ML predictions")
            
            # Extract features from input
            options_data = features.get('options_data')
            technical_indicators = features.get('technical_indicators', {})
            market_data = features.get('market_data', {})
            multi_timeframe_data = features.get('multi_timeframe_data', {})
            
            # Check if we have options data
            if options_data is None or (isinstance(options_data, pd.DataFrame) and options_data.empty):
                logger.warning("No options data provided for prediction")
                
                # Check if we have a symbol to work with
                symbol = features.get('symbol')
                if not symbol:
                    logger.error("No symbol provided for prediction")
                    return self._generate_fallback_prediction(technical_indicators, market_data)
                
                # Try to extract underlying symbol if this is an option
                if self.options_parser.is_option_symbol(symbol):
                    underlying_symbol = self.options_parser.get_underlying_symbol(symbol)
                    logger.info(f"Extracted underlying symbol {underlying_symbol} from option symbol {symbol}")
                else:
                    underlying_symbol = symbol
                
                # Use technical indicators and market data for prediction
                return self._generate_prediction_from_indicators(
                    symbol=underlying_symbol,
                    technical_indicators=technical_indicators,
                    market_data=market_data,
                    multi_timeframe_data=multi_timeframe_data
                )
            
            # If we have options data, proceed with normal prediction
            # Create feature vector
            feature_dict = {
                'price_to_strike_ratio': options_data['underlyingPrice'] / options_data['strikePrice'] if 'strikePrice' in options_data.columns else 1.0,
                'days_to_expiration': options_data['daysToExpiration'] if 'daysToExpiration' in options_data.columns else 30,
                'implied_volatility': options_data['impliedVolatility'] if 'impliedVolatility' in options_data.columns else 0.3,
                'delta': options_data['delta'] if 'delta' in options_data.columns else 0.5,
                'gamma': options_data['gamma'] if 'gamma' in options_data.columns else 0.05,
                'theta': options_data['theta'] if 'theta' in options_data.columns else -0.05,
                'vega': options_data['vega'] if 'vega' in options_data.columns else 0.1,
                'rho': options_data['rho'] if 'rho' in options_data.columns else 0.01,
                'market_trend': market_data.get('market_trend', 0),
                'volatility': market_data.get('volatility', 0),
                'rsi': technical_indicators.get('rsi', 50),
                'macd': technical_indicators.get('macd_histogram', 0)
            }
            
            # Add multi-timeframe indicators if available
            if multi_timeframe_data and 'consolidated_indicators' in multi_timeframe_data:
                consolidated = multi_timeframe_data['consolidated_indicators']
                for key, value in consolidated.items():
                    feature_dict[f'mt_{key}'] = value
            
            # Generate prediction
            # In a real implementation, this would use the trading system's ML models
            # For now, we'll use a simplified approach based on the features
            
            # Calculate a simple prediction based on delta and market trend
            delta_value = feature_dict['delta']
            market_trend = feature_dict['market_trend']
            iv = feature_dict['implied_volatility']
            rsi = feature_dict['rsi']
            
            # Predicted return calculation (simplified)
            predicted_return = delta_value * 0.1 + market_trend * 0.05
            
            # Adjust based on RSI (overbought/oversold)
            if rsi > 70:  # Overbought
                predicted_return -= 0.02
            elif rsi < 30:  # Oversold
                predicted_return += 0.02
                
            # Confidence calculation
            confidence = 0.5 + abs(delta_value) * 0.2 + (1 - iv) * 0.2
            confidence = min(max(confidence, 0.3), 0.9)  # Bound between 0.3 and 0.9
            
            # Risk score calculation
            risk_score = iv * 0.5 + abs(delta_value) * 0.3 + abs(feature_dict['gamma']) * 0.2
            risk_score = min(max(risk_score, 0.1), 0.9)  # Bound between 0.1 and 0.9
            
            # Probability of profit (simplified)
            probability_of_profit = 0.5 + delta_value * (0.5 if delta_value > 0 else -0.5)
            probability_of_profit = min(max(probability_of_profit, 0.05), 0.95)
            
            result = {
                'prediction': predicted_return,
                'confidence': confidence,
                'risk_score': risk_score,
                'probability_of_profit': probability_of_profit
            }
            
            logger.info(f"Generated prediction with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            return self._generate_fallback_prediction(technical_indicators, market_data)
    
    def _generate_prediction_from_indicators(self, symbol, technical_indicators, market_data, multi_timeframe_data):
        """
        Generate a prediction based on technical indicators and market data when options data is not available.
        
        Parameters:
        -----------
        symbol : str
            Underlying symbol
        technical_indicators : dict
            Technical indicators
        market_data : dict
            Market context data
        multi_timeframe_data : dict
            Data from multiple timeframes
            
        Returns:
        --------
        dict
            Prediction results
        """
        logger.info(f"Generating prediction from indicators for {symbol}")
        
        # Default values
        trend = 0
        volatility = 0.5
        momentum = 0
        
        # Extract signals from multi-timeframe data if available
        if multi_timeframe_data and 'signals' in multi_timeframe_data:
            signals = multi_timeframe_data['signals']
            trend = signals.get('trend', 0)
            volatility = signals.get('volatility', 0.5)
            momentum = signals.get('momentum', 0)
        else:
            # Extract from individual indicators
            rsi = technical_indicators.get('rsi', 50)
            macd = technical_indicators.get('macd', 0)
            macd_histogram = technical_indicators.get('macd_histogram', 0)
            
            # Calculate trend
            if macd > 0 and macd_histogram > 0:
                trend = 0.5
            elif macd < 0 and macd_histogram < 0:
                trend = -0.5
                
            # Calculate momentum
            if rsi > 70:
                momentum = 0.8
            elif rsi < 30:
                momentum = -0.8
            else:
                momentum = (rsi - 50) / 25  # Scale to -1 to 1
        
        # Calculate prediction
        predicted_return = trend * 0.05 + momentum * 0.03
        
        # Adjust for market data if available
        if market_data:
            market_trend = market_data.get('market_trend', 0)
            predicted_return += market_trend * 0.02
        
        # Calculate confidence based on available data quality
        confidence = 0.4  # Base confidence is lower without options data
        if multi_timeframe_data and 'signals' in multi_timeframe_data:
            confidence += 0.2  # Higher confidence with multi-timeframe data
        
        # Risk score based on volatility
        risk_score = volatility
        
        # Probability of profit
        probability_of_profit = 0.5 + (trend + momentum) / 4
        probability_of_profit = min(max(probability_of_profit, 0.05), 0.95)
        
        result = {
            'prediction': predicted_return,
            'confidence': confidence,
            'risk_score': risk_score,
            'probability_of_profit': probability_of_profit,
            'source': 'technical_indicators'  # Indicate this is based on indicators, not options data
        }
        
        logger.info(f"Generated indicator-based prediction with confidence {confidence:.2f}")
        return result
    
    def _generate_fallback_prediction(self, technical_indicators=None, market_data=None):
        """
        Generate a fallback prediction when no data is available.
        
        Parameters:
        -----------
        technical_indicators : dict, optional
            Technical indicators if available
        market_data : dict, optional
            Market context data if available
            
        Returns:
        --------
        dict
            Basic prediction results with low confidence
        """
        logger.warning("Generating fallback prediction due to missing data")
        
        # Use any available indicators
        rsi = 50
        if technical_indicators and 'rsi' in technical_indicators:
            rsi = technical_indicators['rsi']
        
        # Very basic prediction
        if rsi > 70:
            prediction = -0.01  # Slightly bearish if overbought
        elif rsi < 30:
            prediction = 0.01   # Slightly bullish if oversold
        else:
            prediction = 0      # Neutral
            
        return {
            'prediction': prediction,
            'confidence': 0.2,  # Very low confidence
            'risk_score': 0.5,  # Moderate risk
            'probability_of_profit': 0.5,  # 50/50
            'source': 'fallback'  # Indicate this is a fallback prediction
        }
    
    def process_recommendation(self, recommendation):
        """
        Process a recommendation through risk management system.
        This method is called by the EnhancedRecommendationEngine.
        
        Parameters:
        -----------
        recommendation : dict
            Recommendation data
            
        Returns:
        --------
        dict
            Processed recommendation with risk management details
        """
        try:
            logger.info(f"Processing recommendation for {recommendation.get('symbol', 'unknown')}")
            
            # Extract recommendation details
            symbol = recommendation.get('symbol', '')
            prediction = recommendation.get('prediction', 0)
            confidence = recommendation.get('confidence', {}).get('score', 0.5)
            
            # Apply risk management
            position_sizing = self._calculate_position_sizing(symbol, prediction, confidence)
            exit_points = self._calculate_exit_points(symbol, prediction, confidence)
            
            # Combine into result
            result = {
                'symbol': symbol,
                'original_recommendation': recommendation,
                'risk_management': {
                    'position_sizing': position_sizing,
                    'exit_points': exit_points,
                    'risk_reward_ratio': exit_points.get('risk_reward_ratio', 1.0)
                }
            }
            
            logger.info(f"Processed recommendation with risk-reward ratio {exit_points.get('risk_reward_ratio', 1.0):.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing recommendation: {str(e)}")
            return recommendation
    
    def _calculate_position_sizing(self, symbol, prediction, confidence):
        """
        Calculate position sizing based on prediction and confidence.
        
        Parameters:
        -----------
        symbol : str
            Symbol
        prediction : float
            Predicted return
        confidence : float
            Confidence score
            
        Returns:
        --------
        dict
            Position sizing details
        """
        # Default values
        account_size = 100000  # Example account size
        max_risk_percent = 0.02  # 2% max risk per trade
        
        # Adjust risk based on confidence
        risk_percent = max_risk_percent * confidence
        
        # Calculate position size
        risk_amount = account_size * risk_percent
        
        # Simplified calculation for number of contracts
        # In a real implementation, this would consider option price, delta, etc.
        contract_price = 3.50  # Example option price
        max_contracts = int(risk_amount / (contract_price * 100))
        recommended_contracts = max(1, max_contracts)
        
        return {
            'account_size': account_size,
            'risk_percentage': risk_percent,
            'total_risk': risk_amount,
            'max_contracts': max_contracts,
            'recommended_contracts': recommended_contracts
        }
    
    def _calculate_exit_points(self, symbol, prediction, confidence):
        """
        Calculate exit points (stop loss and take profit) based on prediction.
        
        Parameters:
        -----------
        symbol : str
            Symbol
        prediction : float
            Predicted return
        confidence : float
            Confidence score
            
        Returns:
        --------
        dict
            Exit points details
        """
        # Default values
        entry_price = 3.50  # Example option price
        
        # Calculate stop loss and take profit based on prediction and confidence
        stop_loss_percent = 0.3 * (1 - confidence)  # Higher confidence = tighter stop
        take_profit_percent = abs(prediction) * 2
        
        # Ensure minimum values
        stop_loss_percent = max(stop_loss_percent, 0.15)
        take_profit_percent = max(take_profit_percent, 0.3)
        
        # Calculate actual prices
        stop_loss = entry_price * (1 - stop_loss_percent)
        take_profit = entry_price * (1 + take_profit_percent)
        
        # Calculate risk-reward ratio
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward_ratio = reward / risk if risk > 0 else 1.0
        
        return {
            'entry_price': entry_price,
            'initial_stop_loss': stop_loss,
            'initial_take_profit': take_profit,
            'stop_loss_percent': stop_loss_percent,
            'take_profit_percent': take_profit_percent,
            'risk_reward_ratio': risk_reward_ratio,
            'final_stop_loss': stop_loss,
            'final_take_profit': take_profit
        }
