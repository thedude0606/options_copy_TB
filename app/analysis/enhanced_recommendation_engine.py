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

# Import enhanced logging
from app.utils.enhanced_logging import EnhancedLogger, ErrorHandler

def safe_get_last(series_or_df):
    """
    Safely get the last element of a Series or DataFrame, returning None if empty.
    
    Args:
        series_or_df: pandas Series or DataFrame to get last element from
        
    Returns:
        Last element or None if empty/None
    """
    if series_or_df is None or (hasattr(series_or_df, 'empty') and series_or_df.empty):
        return None
    try:
        return series_or_df.iloc[-1]
    except (IndexError, KeyError):
        return None

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
        
        # Initialize enhanced logging
        self.logger = EnhancedLogger('enhanced_recommendation_engine')
        if debug:
            self.logger.logger.setLevel(logging.DEBUG)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(logger=self.logger)
        
        # Initialize the enhanced ML integration
        self.ml_integration = EnhancedMLIntegration(config_path=ml_config_path)
        
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
            self.logger.error(f"Error generating enhanced recommendations: {str(e)}", exc_info=e)
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
                
                # Get multi-timeframe data for the underlying symbol
                multi_timeframe_data = self._get_multi_timeframe_data(symbol)
                
                # Get technical indicators
                indicators = self._get_technical_indicators(symbol, lookback_days=30)
                
                # Create feature vector
                features = {
                    'symbol': symbol,
                    'option_data': rec,
                    'technical_indicators': indicators,
                    'market_data': self._get_market_context(symbol),
                    'multi_timeframe_data': multi_timeframe_data
                }
                
                # Get ML prediction
                ml_result = self.error_handler.safe_execute(
                    self.ml_integration.predict,
                    features,
                    default_return={'confidence': 0.5, 'prediction': 0, 'risk_score': 0.5}
                )
                
                # Enhance recommendation with ML insights
                if ml_result and 'prediction' in ml_result:
                    rec['mlConfidence'] = ml_result.get('confidence', 0.5)
                    rec['mlPredictedReturn'] = ml_result.get('prediction', 0)
                    rec['mlRiskScore'] = ml_result.get('risk_score', 0.5)
                    rec['mlSource'] = ml_result.get('source', 'ml_model')
                    
                    # Adjust recommendation score based on ML
                    base_score = rec.get('score', 0)
                    ml_score = ml_result.get('confidence', 0.5) * 100
                    rec['score'] = (base_score * 0.6) + (ml_score * 0.4)  # Weighted combination
                
                enhanced_recs.append(rec)
            
            # Convert back to DataFrame
            return pd.DataFrame(enhanced_recs)
            
        except Exception as e:
            self.logger.error(f"Error applying ML enhancements: {str(e)}", exc_info=e)
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
                enhanced_rec = self.error_handler.safe_execute(
                    self.ml_integration.process_recommendation,
                    recommendation,
                    default_return=recommendation
                )
                
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
            self.logger.error(f"Error applying risk management: {str(e)}", exc_info=e)
            return recommendations
    
    def _get_multi_timeframe_data(self, symbol):
        """
        Get multi-timeframe data for a symbol.
        
        Args:
            symbol (str): Symbol to get data for
            
        Returns:
            dict: Multi-timeframe data and analysis
        """
        try:
            from app.indicators.multi_timeframe_analyzer import MultiTimeframeAnalyzer
            
            # Get multi-timeframe data
            timeframe_data = self.data_collector.get_multi_timeframe_data(symbol)
            
            # Initialize analyzer
            analyzer = MultiTimeframeAnalyzer()
            
            # Analyze data
            analysis_result = analyzer.analyze(timeframe_data)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error getting multi-timeframe data for {symbol}: {str(e)}", exc_info=e)
            return {}
    
    def _get_technical_indicators(self, symbol, lookback_days=30):
        """
        Get technical indicators for a symbol.
        
        Args:
            symbol (str): Symbol to get indicators for
            lookback_days (int): Number of days to look back
            
        Returns:
            dict: Technical indicators
        """
        try:
            # Get historical data
            historical_data = self.data_collector.get_historical_data(
                symbol=symbol,
                period_type='month',
                period=1,
                frequency_type='daily',
                frequency=1
            )
            
            if historical_data.empty:
                self.logger.warning(f"No historical data available for {symbol}")
                return {}
            
            # Calculate indicators
            from app.indicators.technical_indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in historical_data.columns]
            
            if missing_columns:
                # Try to map common column names
                column_mapping = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                for orig, new in column_mapping.items():
                    if orig in historical_data.columns and new not in historical_data.columns:
                        historical_data[new] = historical_data[orig]
            
            # Calculate indicators
            result = {}
            
            # RSI
            if 'close' in historical_data.columns:
                result['rsi'] = safe_get_last(indicators.rsi(historical_data['close']))
            
            # MACD
            if 'close' in historical_data.columns:
                macd_result = indicators.macd(historical_data['close'])
                # Check if macd_result is a tuple (from calculate_macd) or a DataFrame (from macd method)
                if isinstance(macd_result, tuple) and len(macd_result) == 3:
                    # Handle tuple return (macd_line, signal_line, histogram)
                    result['macd'] = safe_get_last(macd_result[0])
                    result['macd_signal'] = safe_get_last(macd_result[1])
                    result['macd_histogram'] = safe_get_last(macd_result[2])
                elif isinstance(macd_result, pd.DataFrame) and all(k in macd_result for k in ['macd', 'signal', 'histogram']):
                    # Handle DataFrame return with expected columns
                    result['macd'] = safe_get_last(macd_result['macd'])
                    result['macd_signal'] = safe_get_last(macd_result['signal'])
                    result['macd_histogram'] = safe_get_last(macd_result['histogram'])
                else:
                    # Log the issue but continue processing
                    self.logger.warning(f"Unexpected MACD result format: {type(macd_result)}")
            
            # Bollinger Bands
            if 'close' in historical_data.columns:
                bb_result = indicators.bollinger_bands(historical_data['close'])
                # Check if bb_result is a tuple or a DataFrame
                if isinstance(bb_result, tuple) and len(bb_result) >= 4:
                    # Handle tuple return (middle_band, upper_band, lower_band, bandwidth)
                    result['bb_middle'] = safe_get_last(bb_result[0])
                    result['bb_upper'] = safe_get_last(bb_result[1])
                    result['bb_lower'] = safe_get_last(bb_result[2])
                    result['bb_width'] = safe_get_last(bb_result[3])
                elif isinstance(bb_result, pd.DataFrame) and all(k in bb_result for k in ['middle', 'upper', 'lower', 'width']):
                    # Handle DataFrame return with expected columns
                    result['bb_middle'] = safe_get_last(bb_result['middle'])
                    result['bb_upper'] = safe_get_last(bb_result['upper'])
                    result['bb_lower'] = safe_get_last(bb_result['lower'])
                    result['bb_width'] = safe_get_last(bb_result['width'])
                else:
                    # Log the issue but continue processing
                    self.logger.warning(f"Unexpected Bollinger Bands result format: {type(bb_result)}")
            
            # ATR
            if all(col in historical_data.columns for col in ['high', 'low', 'close']):
                result['atr'] = safe_get_last(indicators.atr(
                    historical_data['high'],
                    historical_data['low'],
                    historical_data['close']
                ))
            
            # ADX
            if all(col in historical_data.columns for col in ['high', 'low', 'close']):
                adx_result = indicators.adx(
                    historical_data['high'],
                    historical_data['low'],
                    historical_data['close']
                )
                result['adx'] = safe_get_last(adx_result['adx'])
                result['di_plus'] = safe_get_last(adx_result['di_plus'])
                result['di_minus'] = safe_get_last(adx_result['di_minus'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}", exc_info=e)
            return {}
    
    def _get_market_context(self, symbol):
        """
        Get market context data.
        
        Args:
            symbol (str): The symbol to get market context for (may be unused)
            
        Returns:
            dict: Market context data
        """
        try:
            # Get SPY data for market trend
            spy_data = self.error_handler.safe_execute(
                self.data_collector.get_historical_data,
                symbol='SPY',
                period_type='month',
                period=1,
                frequency_type='daily',
                frequency=1,
                default_return=pd.DataFrame()
            )
            
            # Get VIX data for volatility
            vix_data = self.error_handler.safe_execute(
                self.data_collector.get_historical_data,
                symbol='VIX',
                period_type='month',
                period=1,
                frequency_type='daily',
                frequency=1,
                default_return=pd.DataFrame()
            )
            
            result = {}
            
            # Calculate market trend
            if not spy_data.empty and 'close' in spy_data.columns:
                # Calculate 5-day and 20-day moving averages
                spy_data['ma5'] = spy_data['close'].rolling(window=5).mean()
                spy_data['ma20'] = spy_data['close'].rolling(window=20).mean()
                
                # Get latest values
                latest = spy_data.iloc[-1]
                
                # Calculate trend
                if latest['ma5'] > latest['ma20']:
                    # Bullish trend
                    trend_strength = (latest['ma5'] / latest['ma20'] - 1) * 10
                    result['market_trend'] = min(trend_strength, 1.0)
                else:
                    # Bearish trend
                    trend_strength = (1 - latest['ma5'] / latest['ma20']) * 10
                    result['market_trend'] = -min(trend_strength, 1.0)
            
            # Calculate volatility
            if not vix_data.empty and 'close' in vix_data.columns:
                latest_vix = vix_data['close'].iloc[-1]
                
                # Normalize VIX (0-1 scale)
                # VIX below 15 is low volatility, above 30 is high
                if latest_vix < 15:
                    result['volatility'] = 0.2
                elif latest_vix > 30:
                    result['volatility'] = 0.8
                else:
                    result['volatility'] = 0.2 + (latest_vix - 15) / 15 * 0.6
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}", exc_info=e)
            return {}
