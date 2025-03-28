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
        # Initialize the original recommendation engine (without debug parameter)
        super().__init__(data_collector)
        
        # Store debug flag
        self.debug = debug
        
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
    
    def generate_recommendations(self, symbol=None, lookback_days=30, confidence_threshold=0.6, strategy_types=None, symbols=None, strategy_type='all', max_recommendations=10):
        """
        Generate enhanced options trading recommendations using ML models.
        
        Args:
            symbol (str, optional): Stock symbol to generate recommendations for (single symbol)
            lookback_days (int, optional): Number of days to look back for historical data
            confidence_threshold (float, optional): Minimum confidence threshold for recommendations
            strategy_types (list, optional): List of strategy types to consider
            symbols (list, optional): List of symbols to generate recommendations for (multiple symbols)
            strategy_type (str, optional): Type of strategy to recommend
            max_recommendations (int, optional): Maximum number of recommendations to return
            
        Returns:
            pandas.DataFrame: Enhanced recommendations
        """
        try:
            # Handle single symbol case (from recommendations_tab.py)
            if symbol and not symbols:
                symbols = [symbol]
                self.logger.info(f"Converting single symbol {symbol} to symbols list")
            
            # Convert strategy_types to strategy_type if needed
            if strategy_types and not strategy_type or strategy_type == 'all':
                strategy_type = strategy_types[0] if isinstance(strategy_types, list) and strategy_types else strategy_types
                self.logger.info(f"Using strategy_type: {strategy_type}")
            
            # Log recommendation generation start
            self.logger.info(f"Generating enhanced recommendations for {len(symbols) if symbols else 'all'} symbols")
            
            # Get base recommendations from parent class
            # Ensure lookback_days is not None
            lookback_days_value = 30 if lookback_days is None else lookback_days
            
            # Convert parameters to match parent class expectations
            # Original expects: symbol, lookback_days, confidence_threshold, strategy_types
            symbol_value = symbols[0] if symbols and len(symbols) > 0 else None
            strategy_types_value = [strategy_type] if strategy_type and strategy_type != 'all' else None
            
            recommendations = super().generate_recommendations(
                symbol_value,
                lookback_days_value,
                confidence_threshold,
                strategy_types_value
            )
            
            # If no recommendations or ML integration is not available, return base recommendations
            if recommendations.empty or not hasattr(self, 'ml_integration'):
                return recommendations
            
            # Apply ML enhancements
            enhanced_recommendations = self._apply_ml_enhancements(recommendations)
            
            # Apply risk management
            risk_managed_recommendations = self._apply_risk_management(enhanced_recommendations)
            
            # Log recommendation generation completion
            self.logger.info(f"Generated {len(risk_managed_recommendations)} enhanced recommendations")
            
            return risk_managed_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced recommendations: {str(e)}")
            # Fall back to original recommendation engine
            self.logger.info("Falling back to original recommendation engine")
            lookback_days_value = 30 if lookback_days is None else lookback_days
            
            # Convert parameters to match parent class expectations
            # Original expects: symbol, lookback_days, confidence_threshold, strategy_types
            symbol_value = symbols[0] if symbols and len(symbols) > 0 else None
            strategy_types_value = [strategy_type] if strategy_type and strategy_type != 'all' else None
            
            return super().generate_recommendations(
                symbol_value,
                lookback_days_value,
                confidence_threshold,
                strategy_types_value
            )
    
    def _apply_ml_enhancements(self, recommendations):
        """
        Apply machine learning enhancements to recommendations.
        
        Args:
            recommendations (pandas.DataFrame): Base recommendations
            
        Returns:
            pandas.DataFrame: Enhanced recommendations
        """
        try:
            if recommendations.empty:
                return recommendations
            
            # Convert recommendations to list of dictionaries
            rec_list = recommendations.to_dict('records')
            
            # Process each recommendation through ML models
            enhanced_recs = []
            for rec in rec_list:
                # Get symbol and option data
                symbol = rec.get('symbol', '')
                
                # Get technical indicators
                indicators = self._get_technical_indicators(symbol, lookback_days=30)
                
                # Create feature vector
                features = {
                    'option_data': rec,
                    'technical_indicators': indicators,
                    'market_data': self._get_market_context()
                }
                
                # Get ML prediction
                ml_result = self.ml_integration.predict(features)
                
                # Enhance recommendation with ML insights
                if ml_result and 'prediction' in ml_result:
                    rec['mlConfidence'] = ml_result.get('confidence', 0.5)
                    rec['mlPredictedReturn'] = ml_result.get('prediction', 0)
                    rec['mlRiskScore'] = ml_result.get('risk_score', 0.5)
                    
                    # Adjust recommendation score based on ML
                    base_score = rec.get('score', 0)
                    ml_score = ml_result.get('confidence', 0.5) * 100
                    rec['score'] = (base_score * 0.6) + (ml_score * 0.4)  # Weighted combination
                
                enhanced_recs.append(rec)
            
            # Convert back to DataFrame
            return pd.DataFrame(enhanced_recs)
            
        except Exception as e:
            self.logger.error(f"Error applying ML enhancements: {str(e)}")
            return recommendations
    
    def _apply_risk_management(self, recommendations):
        """
        Apply risk management to recommendations.
        
        Args:
            recommendations (pandas.DataFrame): Enhanced recommendations
            
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
    
    def _get_market_context(self):
        """
        Get current market context data.
        
        Returns:
            dict: Market context data
        """
        try:
            # Get market indices
            spy_data = self.data_collector.get_quote('SPY')
            vix_data = self.data_collector.get_quote('VIX')
            
            market_context = {
                'market_trend': 0,  # Neutral by default
                'volatility': 0,
                'sector_performance': {}
            }
            
            # Set market trend based on SPY
            if spy_data and 'lastPrice' in spy_data and 'openPrice' in spy_data:
                if spy_data['lastPrice'] > spy_data['openPrice']:
                    market_context['market_trend'] = 1  # Bullish
                elif spy_data['lastPrice'] < spy_data['openPrice']:
                    market_context['market_trend'] = -1  # Bearish
            
            # Set volatility based on VIX
            if vix_data and 'lastPrice' in vix_data:
                market_context['volatility'] = vix_data['lastPrice']
            
            return market_context
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}")
            return {
                'market_trend': 0,
                'volatility': 0,
                'sector_performance': {}
            }
