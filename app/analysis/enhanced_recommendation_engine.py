"""
Enhanced recommendation engine that integrates machine learning components.
Extends the original recommendation engine with advanced ML features.
"""
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import original recommendation engine
from app.analysis.recommendation_engine import RecommendationEngine as OriginalRecommendationEngine

# Import enhanced ML components
from app.ml.enhanced.integration import EnhancedMLIntegration

class EnhancedRecommendationEngine(OriginalRecommendationEngine):
    """
    Enhanced recommendation engine that integrates machine learning components.
    Extends the original recommendation engine with advanced ML features.
    """
    def __init__(self, data_collector, ml_config_path=None, debug=False):
        """
        Initialize the enhanced recommendation engine.
        
        Args:
            data_collector: The data collector instance
            ml_config_path (str, optional): Path to ML configuration file
            debug (bool, optional): Whether to enable debug output
        """
        # Initialize the original recommendation engine
        super().__init__(data_collector, debug=debug)
        
        # Initialize the enhanced ML integration
        self.ml_integration = EnhancedMLIntegration(config_path=ml_config_path)
        
        # Set up logging
        self.logger = logging.getLogger('enhanced_recommendation_engine')
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Log initialization
        self.logger.info("Enhanced recommendation engine initialized")
        
        # Create cache directory for ML models if it doesn't exist
        self.ml_cache_dir = os.path.join(self.cache_dir, 'ml_models')
        os.makedirs(self.ml_cache_dir, exist_ok=True)
    
    def generate_recommendations(self, symbol, lookback_days=30, confidence_threshold=0.6):
        """
        Generate options trading recommendations using enhanced ML features.
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int, optional): Number of days to look back for historical data
            confidence_threshold (float, optional): Minimum confidence threshold for recommendations
            
        Returns:
            pandas.DataFrame: DataFrame with recommendations
        """
        if self.debug:
            self.logger.debug(f"Generating enhanced recommendations for {symbol}")
        
        try:
            # Get options data using the original method
            options_data = self._get_options_data(symbol)
            
            if options_data is None or options_data.empty:
                self.logger.warning(f"No options data available for {symbol}")
                return pd.DataFrame()
            
            # Get underlying price
            underlying_price = self.get_underlying_price(symbol)
            if underlying_price is None:
                self.logger.warning(f"Could not retrieve underlying price for {symbol}")
                return pd.DataFrame()
            
            # Get technical indicators
            indicators = self._get_technical_indicators(symbol, lookback_days)
            
            # Prepare data for ML processing
            ml_ready_data = self._prepare_data_for_ml(options_data, indicators, underlying_price, symbol)
            
            if ml_ready_data is None or ml_ready_data.empty:
                self.logger.warning(f"Failed to prepare data for ML processing for {symbol}")
                return pd.DataFrame()
            
            # Process data through ML pipeline
            if self.debug:
                self.logger.debug(f"Processing {len(ml_ready_data)} options through ML pipeline")
            
            # Generate ML-enhanced recommendations
            ml_recommendations = self._generate_ml_recommendations(ml_ready_data, confidence_threshold)
            
            if ml_recommendations is None or ml_recommendations.empty:
                self.logger.warning(f"No ML recommendations generated for {symbol}")
                
                # Fall back to original recommendation method
                self.logger.info(f"Falling back to original recommendation method for {symbol}")
                return super().generate_recommendations(symbol, lookback_days, confidence_threshold)
            
            # Enhance recommendations with risk management
            enhanced_recommendations = self._add_risk_management(ml_recommendations)
            
            if self.debug:
                self.logger.debug(f"Generated {len(enhanced_recommendations)} enhanced recommendations for {symbol}")
            
            return enhanced_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced recommendations: {str(e)}")
            
            # Fall back to original recommendation method
            self.logger.info(f"Falling back to original recommendation method due to error")
            return super().generate_recommendations(symbol, lookback_days, confidence_threshold)
    
    def _prepare_data_for_ml(self, options_data, indicators, underlying_price, symbol):
        """
        Prepare options data for ML processing.
        
        Args:
            options_data (pandas.DataFrame): Options data
            indicators (dict): Technical indicators
            underlying_price (float): Current price of the underlying asset
            symbol (str): The stock symbol
            
        Returns:
            pandas.DataFrame: Prepared data for ML processing
        """
        try:
            if self.debug:
                self.logger.debug(f"Preparing data for ML processing for {symbol}")
            
            # Make a copy to avoid modifying the original
            ml_data = options_data.copy()
            
            # Add symbol column
            ml_data['symbol'] = symbol
            
            # Add underlying price
            ml_data['underlyingPrice'] = underlying_price
            
            # Add technical indicators as features
            for indicator, value in indicators.items():
                ml_data[f'indicator_{indicator}'] = value
            
            # Calculate days to expiration
            if 'expirationDate' in ml_data.columns:
                today = datetime.now().date()
                ml_data['daysToExpiration'] = (pd.to_datetime(ml_data['expirationDate']).dt.date - today).dt.days
            
            # Calculate moneyness
            if 'strike' in ml_data.columns and underlying_price:
                ml_data['moneyness'] = ml_data['strike'] / underlying_price
            
            # Calculate mid price
            if 'bid' in ml_data.columns and 'ask' in ml_data.columns:
                ml_data['midPrice'] = (ml_data['bid'] + ml_data['ask']) / 2
            
            # Calculate bid-ask spread
            if 'bid' in ml_data.columns and 'ask' in ml_data.columns:
                ml_data['bidAskSpread'] = ml_data['ask'] - ml_data['bid']
                ml_data['bidAskSpreadPct'] = ml_data['bidAskSpread'] / ml_data['midPrice']
            
            return ml_data
            
        except Exception as e:
            self.logger.error(f"Error preparing data for ML: {str(e)}")
            return None
    
    def _generate_ml_recommendations(self, ml_data, confidence_threshold):
        """
        Generate recommendations using ML models.
        
        Args:
            ml_data (pandas.DataFrame): Prepared data for ML processing
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            pandas.DataFrame: ML-enhanced recommendations
        """
        try:
            if self.debug:
                self.logger.debug(f"Generating ML recommendations for {len(ml_data)} options")
            
            # Generate predictions using ML integration
            prediction_result = self.ml_integration.generate_predictions(ml_data)
            
            if prediction_result is None:
                self.logger.warning("ML prediction failed")
                return pd.DataFrame()
            
            # Extract predictions
            predictions = prediction_result.get('predictions', [])
            confidence_scores = prediction_result.get('confidence_scores', [])
            
            if len(predictions) == 0:
                self.logger.warning("No predictions generated")
                return pd.DataFrame()
            
            # Add predictions to data
            ml_data['ml_prediction'] = predictions
            ml_data['ml_confidence'] = confidence_scores
            
            # Filter by confidence threshold
            recommendations = ml_data[ml_data['ml_confidence'] >= confidence_threshold]
            
            # If no recommendations meet the threshold, get top 5
            if recommendations.empty:
                recommendations = ml_data.nlargest(5, 'ml_confidence')
            
            # Sort by confidence
            recommendations = recommendations.sort_values('ml_confidence', ascending=False)
            
            # Format recommendations to match original format
            formatted_recommendations = self._format_ml_recommendations(recommendations)
            
            return formatted_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating ML recommendations: {str(e)}")
            return pd.DataFrame()
    
    def _format_ml_recommendations(self, ml_recommendations):
        """
        Format ML recommendations to match the original format.
        
        Args:
            ml_recommendations (pandas.DataFrame): ML-enhanced recommendations
            
        Returns:
            pandas.DataFrame: Formatted recommendations
        """
        try:
            # Create a new DataFrame with the required columns
            formatted = pd.DataFrame()
            
            # Map columns from ML recommendations to original format
            column_mapping = {
                'symbol': 'symbol',
                'putCall': 'optionType',
                'strike': 'strike',
                'expirationDate': 'expirationDate',
                'bid': 'bid',
                'ask': 'ask',
                'midPrice': 'price',
                'underlyingPrice': 'underlyingPrice',
                'delta': 'delta',
                'gamma': 'gamma',
                'theta': 'theta',
                'vega': 'vega',
                'rho': 'rho',
                'daysToExpiration': 'daysToExpiration',
                'ml_prediction': 'potentialReturn',
                'ml_confidence': 'confidence'
            }
            
            # Copy mapped columns
            for ml_col, orig_col in column_mapping.items():
                if ml_col in ml_recommendations.columns:
                    formatted[orig_col] = ml_recommendations[ml_col]
            
            # Add additional columns
            formatted['score'] = (formatted['confidence'] * 100).astype(int)
            
            # Add market direction and strategy
            formatted['marketDirection'] = 'BULLISH'  # Default
            formatted['strategy'] = 'ML_ENHANCED'  # Mark as ML-enhanced
            
            # Add signal details
            formatted['signalDetails'] = 'Generated by enhanced ML model'
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting ML recommendations: {str(e)}")
            return pd.DataFrame()
    
    def _add_risk_management(self, recommendations):
        """
        Add risk management features to recommendations.
        
        Args:
            recommendations (pandas.DataFrame): ML recommendations
            
        Returns:
            pandas.DataFrame: Recommendations with risk management
        """
        try:
            if recommendations.empty:
                return recommendations
            
            # Convert recommendations to list of dictionaries
            rec_list = recommendations.to_dict('records')
            
            # Process each recommendation through risk management
            enhanced_recs = []
            for rec in rec_list:
                # Create recommendation object for risk management
                recommendation = {
                    'symbol': rec.get('symbol', ''),
                    'option_data': rec,
                    'prediction': rec.get('potentialReturn', 0),
                    'confidence': {
                        'score': rec.get('confidence', 0),
                        'model': 'enhanced_ml'
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Process through risk management
                enhanced_rec = self.ml_integration.process_recommendation(recommendation)
                
                # Extract risk management details
                if enhanced_rec and 'risk_management' in enhanced_rec:
                    risk_mgmt = enhanced_rec['risk_management']
                    
                    # Add position sizing
                    if 'position_sizing' in risk_mgmt:
                        position = risk_mgmt['position_sizing']
                        rec['recommendedContracts'] = position.get('recommended_contracts', 1)
                        rec['maxRiskAmount'] = position.get('total_risk', 0)
                        rec['riskPercentage'] = position.get('risk_percentage', 0)
                    
                    # Add exit points
                    if 'exit_points' in risk_mgmt:
                        exits = risk_mgmt['exit_points']
                        rec['stopLoss'] = exits.get('final_stop_loss', 0)
                        rec['takeProfit'] = exits.get('final_take_profit', 0)
                        rec['riskRewardRatio'] = exits.get('risk_reward_ratio', 0)
                
                enhanced_recs.append(rec)
            
            # Convert back to DataFrame
            return pd.DataFrame(enhanced_recs)
            
        except Exception as e:
            self.logger.error(f"Error adding risk management: {str(e)}")
            return recommendations
    
    def _get_technical_indicators(self, symbol, lookback_days):
        """
        Get technical indicators for the symbol.
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back
            
        Returns:
            dict: Dictionary of technical indicators
        """
        try:
            # Get historical data
            historical_data = self.data_collector.get_historical_data(
                symbol, 
                period_type="month", 
                period=1, 
                frequency_type="daily", 
                frequency=1
            )
            
            if historical_data is None or len(historical_data) == 0:
                self.logger.warning(f"No historical data available for {symbol}")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Calculate indicators
            indicators = {}
            
            # RSI
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Moving Averages
            if len(df) >= 50:
                indicators['sma20'] = df['close'].rolling(window=20).mean().iloc[-1]
                indicators['sma50'] = df['close'].rolling(window=50).mean().iloc[-1]
                indicators['ma_trend'] = 1 if indicators['sma20'] > indicators['sma50'] else -1
            
            # Volatility
            if len(df) >= 20:
                indicators['volatility'] = df['close'].pct_change().rolling(window=20).std().iloc[-1]
            
            # Trend
            if len(df) >= 10:
                indicators['price_trend'] = 1 if df['close'].iloc[-1] > df['close'].iloc[-10] else -1
            
            # Volume trend
            if 'volume' in df.columns and len(df) >= 10:
                avg_volume = df['volume'].rolling(window=10).mean().iloc[-1]
                indicators['volume_trend'] = 1 if df['volume'].iloc[-1] > avg_volume else -1
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}
