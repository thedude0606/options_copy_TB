"""
Multi-Timeframe Analyzer Module

This module provides functionality for analyzing data across multiple timeframes
and generating consolidated indicators and signals.
"""

import pandas as pd
import numpy as np
import logging
from app.indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger('multi_timeframe_analyzer')

class MultiTimeframeAnalyzer:
    """
    Analyzer for processing data across multiple timeframes and generating
    consolidated indicators and signals.
    """
    
    def __init__(self, timeframe_weights=None):
        """
        Initialize the multi-timeframe analyzer.
        
        Args:
            timeframe_weights (dict, optional): Dictionary of weights for each timeframe
                                               Default weights prioritize shorter timeframes
        """
        # Default weights if none provided
        self.timeframe_weights = timeframe_weights or {
            'daily': 0.5,
            'weekly': 0.3,
            'monthly': 0.2
        }
        
        # Normalize weights to ensure they sum to 1
        weight_sum = sum(self.timeframe_weights.values())
        if weight_sum != 1.0:
            for key in self.timeframe_weights:
                self.timeframe_weights[key] /= weight_sum
                
        # Initialize technical indicators calculator
        self.indicators = TechnicalIndicators()
        
        logger.info(f"MultiTimeframeAnalyzer initialized with weights: {self.timeframe_weights}")
        
    def analyze(self, timeframe_data, include_indicators=None):
        """
        Analyze data across multiple timeframes and generate consolidated indicators.
        
        Args:
            timeframe_data (dict): Dictionary of DataFrames, keyed by timeframe name
            include_indicators (list, optional): List of indicators to include
                                               If None, includes all available indicators
                                               
        Returns:
            dict: Consolidated indicators and signals
        """
        if not timeframe_data:
            logger.warning("No timeframe data provided for analysis")
            return {}
            
        # Default indicators if none specified
        if include_indicators is None:
            include_indicators = [
                'rsi', 'macd', 'bollinger_bands', 'atr', 'adx',
                'stochastic', 'obv', 'cmf', 'mfi', 'cci'
            ]
            
        logger.info(f"Analyzing {len(timeframe_data)} timeframes with {len(include_indicators)} indicators")
        
        # Calculate indicators for each timeframe
        timeframe_indicators = {}
        
        for timeframe, data in timeframe_data.items():
            if data.empty:
                logger.warning(f"Empty data for timeframe {timeframe}, skipping")
                continue
                
            # Calculate indicators for this timeframe
            indicators_result = self._calculate_indicators(data, include_indicators)
            timeframe_indicators[timeframe] = indicators_result
            
            logger.info(f"Calculated {len(indicators_result)} indicators for {timeframe} timeframe")
            
        # Consolidate indicators across timeframes
        consolidated = self._consolidate_indicators(timeframe_indicators)
        
        # Generate signals based on consolidated indicators
        signals = self._generate_signals(consolidated)
        
        # Combine everything into final result
        result = {
            'consolidated_indicators': consolidated,
            'signals': signals,
            'timeframe_indicators': timeframe_indicators
        }
        
        return result
        
    def _calculate_indicators(self, data, indicator_list):
        """
        Calculate technical indicators for a single timeframe.
        
        Args:
            data (pd.DataFrame): Price data for a single timeframe
            indicator_list (list): List of indicators to calculate
            
        Returns:
            dict: Calculated indicators
        """
        result = {}
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            # Try to map common column names
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            for orig, new in column_mapping.items():
                if orig in data.columns and new not in data.columns:
                    data[new] = data[orig]
        
        # Calculate each requested indicator
        for indicator in indicator_list:
            try:
                if indicator == 'rsi':
                    result['rsi'] = self.indicators.rsi(data['close'])
                elif indicator == 'macd':
                    macd_result = self.indicators.macd(data['close'])
                    result['macd'] = macd_result['macd']
                    result['macd_signal'] = macd_result['signal']
                    result['macd_histogram'] = macd_result['histogram']
                elif indicator == 'bollinger_bands':
                    bb_result = self.indicators.bollinger_bands(data['close'])
                    result['bb_upper'] = bb_result['upper']
                    result['bb_middle'] = bb_result['middle']
                    result['bb_lower'] = bb_result['lower']
                    result['bb_width'] = bb_result['width']
                elif indicator == 'atr':
                    result['atr'] = self.indicators.atr(data['high'], data['low'], data['close'])
                elif indicator == 'adx':
                    adx_result = self.indicators.adx(data['high'], data['low'], data['close'])
                    result['adx'] = adx_result['adx']
                    result['di_plus'] = adx_result['di_plus']
                    result['di_minus'] = adx_result['di_minus']
                elif indicator == 'stochastic':
                    stoch_result = self.indicators.stochastic(data['high'], data['low'], data['close'])
                    result['stoch_k'] = stoch_result['k']
                    result['stoch_d'] = stoch_result['d']
                elif indicator == 'obv':
                    result['obv'] = self.indicators.obv(data['close'], data['volume'])
                elif indicator == 'cmf':
                    result['cmf'] = self.indicators.cmf(data['high'], data['low'], data['close'], data['volume'])
                elif indicator == 'mfi':
                    result['mfi'] = self.indicators.mfi(data['high'], data['low'], data['close'], data['volume'])
                elif indicator == 'cci':
                    result['cci'] = self.indicators.cci(data['high'], data['low'], data['close'])
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {str(e)}")
                
        return result
        
    def _consolidate_indicators(self, timeframe_indicators):
        """
        Consolidate indicators across multiple timeframes using weighted averages.
        
        Args:
            timeframe_indicators (dict): Dictionary of indicators for each timeframe
            
        Returns:
            dict: Consolidated indicators
        """
        consolidated = {}
        
        # Get all indicator keys from all timeframes
        all_indicators = set()
        for tf_indicators in timeframe_indicators.values():
            all_indicators.update(tf_indicators.keys())
            
        # Consolidate each indicator
        for indicator in all_indicators:
            values = []
            weights = []
            
            for timeframe, indicators in timeframe_indicators.items():
                if indicator in indicators and not pd.isna(indicators[indicator].iloc[-1]):
                    values.append(indicators[indicator].iloc[-1])
                    weights.append(self.timeframe_weights.get(timeframe, 0.1))
                    
            if values and weights:
                # Normalize weights for available timeframes
                weights_sum = sum(weights)
                normalized_weights = [w / weights_sum for w in weights]
                
                # Calculate weighted average
                consolidated[indicator] = np.average(values, weights=normalized_weights)
            
        return consolidated
        
    def _generate_signals(self, consolidated):
        """
        Generate trading signals based on consolidated indicators.
        
        Args:
            consolidated (dict): Consolidated indicators
            
        Returns:
            dict: Trading signals
        """
        signals = {
            'trend': 0,  # -1 (bearish), 0 (neutral), 1 (bullish)
            'strength': 0,  # 0 (weak) to 1 (strong)
            'volatility': 0,  # 0 (low) to 1 (high)
            'momentum': 0,  # -1 (negative) to 1 (positive)
            'overall_signal': 'neutral'  # bearish, neutral, bullish
        }
        
        # Trend signal based on MACD and ADX
        if 'macd' in consolidated and 'macd_histogram' in consolidated:
            if consolidated['macd'] > 0 and consolidated['macd_histogram'] > 0:
                signals['trend'] = 1
            elif consolidated['macd'] < 0 and consolidated['macd_histogram'] < 0:
                signals['trend'] = -1
                
        # Adjust trend based on ADX (trend strength)
        if 'adx' in consolidated:
            adx = consolidated['adx']
            signals['strength'] = min(adx / 50, 1.0)  # Normalize to 0-1
            
            # Strong trend confirmation
            if adx > 25:
                if 'di_plus' in consolidated and 'di_minus' in consolidated:
                    if consolidated['di_plus'] > consolidated['di_minus']:
                        signals['trend'] = max(signals['trend'], 0) + 0.5
                    else:
                        signals['trend'] = min(signals['trend'], 0) - 0.5
        
        # Volatility based on ATR and Bollinger Band width
        if 'atr' in consolidated:
            signals['volatility'] = min(consolidated['atr'] / 5, 1.0)  # Normalize to 0-1
            
        if 'bb_width' in consolidated:
            bb_volatility = min(consolidated['bb_width'] / 0.1, 1.0)  # Normalize to 0-1
            signals['volatility'] = (signals['volatility'] + bb_volatility) / 2
            
        # Momentum signals
        if 'rsi' in consolidated:
            rsi = consolidated['rsi']
            if rsi > 70:
                signals['momentum'] = 1
            elif rsi < 30:
                signals['momentum'] = -1
            else:
                signals['momentum'] = (rsi - 50) / 20  # Scale to -1 to 1
                
        # Adjust momentum with MFI if available
        if 'mfi' in consolidated:
            mfi = consolidated['mfi']
            mfi_signal = (mfi - 50) / 20  # Scale to -1 to 1
            signals['momentum'] = (signals['momentum'] + mfi_signal) / 2
            
        # Overall signal calculation
        signal_value = signals['trend'] * 0.4 + signals['momentum'] * 0.3
        
        if signal_value > 0.3:
            signals['overall_signal'] = 'bullish'
        elif signal_value < -0.3:
            signals['overall_signal'] = 'bearish'
        else:
            signals['overall_signal'] = 'neutral'
            
        return signals
        
    def get_adaptive_lookback(self, data, base_lookback=14, volatility_factor=0.5):
        """
        Calculate an adaptive lookback period based on market volatility.
        
        Args:
            data (pd.DataFrame): Price data
            base_lookback (int): Base lookback period
            volatility_factor (float): How much to adjust for volatility (0-1)
            
        Returns:
            int: Adaptive lookback period
        """
        try:
            # Calculate recent volatility using ATR
            atr = self.indicators.atr(data['high'], data['low'], data['close'], period=14)
            recent_atr = atr.iloc[-1]
            
            # Calculate average price for normalization
            avg_price = data['close'].iloc[-1]
            
            # Normalized volatility (0-1 range)
            norm_volatility = min(recent_atr / (avg_price * 0.1), 1.0)
            
            # Adjust lookback period based on volatility
            # Higher volatility = shorter lookback
            adjustment = int(base_lookback * volatility_factor * (1 - norm_volatility))
            
            # Ensure lookback is within reasonable bounds
            adaptive_lookback = max(5, min(base_lookback - adjustment, 30))
            
            logger.info(f"Adaptive lookback: {adaptive_lookback} (base: {base_lookback}, volatility: {norm_volatility:.2f})")
            return adaptive_lookback
            
        except Exception as e:
            logger.error(f"Error calculating adaptive lookback: {str(e)}")
            return base_lookback
