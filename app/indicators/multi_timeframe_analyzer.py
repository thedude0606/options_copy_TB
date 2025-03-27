"""
Multi-timeframe analyzer module for options recommendation platform.
Integrates candlestick patterns with technical indicators across multiple timeframes.
"""
import pandas as pd
import numpy as np
from app.indicators.patterns.candlestick_patterns import CandlestickPatterns
from app.indicators.technical_indicators import TechnicalIndicators

class MultiTimeframeAnalyzer:
    """
    Class to analyze market data across multiple timeframes
    """
    
    def __init__(self, data_collector):
        """
        Initialize the multi-timeframe analyzer
        
        Args:
            data_collector: DataCollector instance for retrieving market data
        """
        self.data_collector = data_collector
        self.technical_indicators = TechnicalIndicators()
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.timeframe_weights = {
            '1m': 0.1,
            '5m': 0.15,
            '15m': 0.2,
            '1h': 0.25,
            '4h': 0.3
        }
    
    def fetch_multi_timeframe_data(self, symbol, lookback_days=30):
        """
        Fetch historical data for multiple timeframes
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back for historical data
            
        Returns:
            dict: Dictionary with timeframes as keys and DataFrame as values
        """
        multi_timeframe_data = {}
        
        for timeframe in self.timeframes:
            # Convert timeframe to appropriate parameters for data collector
            if timeframe.endswith('m'):
                frequency_type = 'minute'
                frequency = int(timeframe[:-1])
            elif timeframe.endswith('h'):
                frequency_type = 'hour'
                frequency = int(timeframe[:-1])
            else:
                frequency_type = 'day'
                frequency = 1
            
            # Determine appropriate lookback period based on timeframe
            if timeframe == '1m':
                period_days = 1  # 1 day for 1-minute data
            elif timeframe == '5m':
                period_days = 5  # 5 days for 5-minute data
            elif timeframe == '15m':
                period_days = 10  # 10 days for 15-minute data
            elif timeframe == '1h':
                period_days = 20  # 20 days for hourly data
            else:
                period_days = lookback_days  # Full lookback for 4h and daily
            
            # Fetch data from data collector
            try:
                data = self.data_collector.get_historical_data(
                    symbol=symbol,
                    period_type='day',
                    period=period_days,
                    frequency_type=frequency_type,
                    frequency=frequency
                )
                
                if not data.empty:
                    multi_timeframe_data[timeframe] = data
            except Exception as e:
                print(f"Error fetching {timeframe} data for {symbol}: {str(e)}")
        
        return multi_timeframe_data
    
    def analyze_timeframe(self, data, timeframe):
        """
        Analyze a single timeframe's data
        
        Args:
            data (pd.DataFrame): OHLC price data for a specific timeframe
            timeframe (str): Timeframe identifier (e.g., '1m', '5m', '15m', '1h', '4h')
            
        Returns:
            dict: Dictionary with analysis results
        """
        if data.empty:
            return {}
        
        # Set data for technical indicators
        self.technical_indicators.data = data
        
        # Calculate technical indicators
        rsi = self.technical_indicators.rsi()
        macd_data = self.technical_indicators.macd()
        bb_data = self.technical_indicators.bollinger_bands()
        
        # Detect candlestick patterns
        candlestick_results = CandlestickPatterns.analyze_candlestick_patterns(data, lookback=10)
        
        # Get the most recent candlestick pattern if available
        latest_pattern = {}
        latest_sentiment = 'neutral'
        pattern_strength = 0
        
        if candlestick_results:
            latest_result = candlestick_results[-1]
            latest_pattern = latest_result['patterns']
            latest_sentiment = latest_result['sentiment']
            pattern_strength = latest_result['strength']
        
        # Get latest indicator values
        latest_rsi = rsi.iloc[-1] if not rsi.empty else None
        
        latest_macd = None
        latest_macd_signal = None
        latest_macd_hist = None
        if not macd_data.empty:
            latest_macd = macd_data['macd'].iloc[-1]
            latest_macd_signal = macd_data['signal'].iloc[-1]
            latest_macd_hist = macd_data['histogram'].iloc[-1]
        
        latest_bb_middle = None
        latest_bb_upper = None
        latest_bb_lower = None
        if not bb_data.empty:
            latest_bb_middle = bb_data['middle_band'].iloc[-1]
            latest_bb_upper = bb_data['upper_band'].iloc[-1]
            latest_bb_lower = bb_data['lower_band'].iloc[-1]
        
        # Determine indicator signals
        rsi_signal = 'neutral'
        if latest_rsi is not None:
            if latest_rsi < 30:
                rsi_signal = 'bullish'
            elif latest_rsi > 70:
                rsi_signal = 'bearish'
        
        macd_signal = 'neutral'
        if latest_macd is not None and latest_macd_signal is not None:
            if latest_macd > latest_macd_signal:
                macd_signal = 'bullish'
            elif latest_macd < latest_macd_signal:
                macd_signal = 'bearish'
        
        bb_signal = 'neutral'
        if (latest_bb_lower is not None and latest_bb_upper is not None and 
            latest_bb_middle is not None and not data.empty):
            latest_close = data['close'].iloc[-1]
            if latest_close < latest_bb_lower:
                bb_signal = 'bullish'  # Oversold
            elif latest_close > latest_bb_upper:
                bb_signal = 'bearish'  # Overbought
        
        # Compile results
        return {
            'timeframe': timeframe,
            'weight': self.timeframe_weights.get(timeframe, 0.2),
            'candlestick_patterns': latest_pattern,
            'candlestick_sentiment': latest_sentiment,
            'pattern_strength': pattern_strength,
            'rsi': latest_rsi,
            'rsi_signal': rsi_signal,
            'macd': latest_macd,
            'macd_signal': latest_macd_signal,
            'macd_hist': latest_macd_hist,
            'macd_trend': macd_signal,
            'bb_middle': latest_bb_middle,
            'bb_upper': latest_bb_upper,
            'bb_lower': latest_bb_lower,
            'bb_signal': bb_signal
        }
    
    def analyze_multi_timeframe(self, symbol, lookback_days=30):
        """
        Analyze market data across multiple timeframes
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back for historical data
            
        Returns:
            dict: Dictionary with analysis results for each timeframe and combined signals
        """
        # Fetch data for all timeframes
        multi_timeframe_data = self.fetch_multi_timeframe_data(symbol, lookback_days)
        
        if not multi_timeframe_data:
            return {
                'symbol': symbol,
                'timeframes': {},
                'combined_signals': {
                    'bullish': 0,
                    'bearish': 0,
                    'neutral': 0,
                    'overall_sentiment': 'neutral',
                    'confidence': 0,
                    'signal_details': []
                }
            }
        
        # Analyze each timeframe
        timeframe_results = {}
        for timeframe, data in multi_timeframe_data.items():
            # Ensure data is not empty before analysis
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                continue
            timeframe_results[timeframe] = self.analyze_timeframe(data, timeframe)
        
        # Combine signals across timeframes
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        signal_details = []
        
        for timeframe, result in timeframe_results.items():
            if not result:
                continue
                
            weight = result['weight']
            
            # Add candlestick pattern signals
            if result['candlestick_sentiment'] == 'bullish':
                bullish_signals += weight * result['pattern_strength']
                if result['pattern_strength'] > 0:
                    patterns_str = ', '.join([p.replace('_', ' ').title() for p in result['candlestick_patterns'].keys()])
                    signal_details.append(f"{timeframe} Bullish patterns: {patterns_str}")
            elif result['candlestick_sentiment'] == 'bearish':
                bearish_signals += weight * result['pattern_strength']
                if result['pattern_strength'] > 0:
                    patterns_str = ', '.join([p.replace('_', ' ').title() for p in result['candlestick_patterns'].keys()])
                    signal_details.append(f"{timeframe} Bearish patterns: {patterns_str}")
            
            # Add RSI signals
            if result['rsi_signal'] == 'bullish':
                bullish_signals += weight * 0.5
                signal_details.append(f"{timeframe} RSI oversold: {result['rsi']:.1f}")
            elif result['rsi_signal'] == 'bearish':
                bearish_signals += weight * 0.5
                signal_details.append(f"{timeframe} RSI overbought: {result['rsi']:.1f}")
            
            # Add MACD signals
            if result['macd_trend'] == 'bullish':
                bullish_signals += weight * 0.6
                signal_details.append(f"{timeframe} MACD bullish crossover")
            elif result['macd_trend'] == 'bearish':
                bearish_signals += weight * 0.6
                signal_details.append(f"{timeframe} MACD bearish crossover")
            
            # Add Bollinger Bands signals
            if result['bb_signal'] == 'bullish':
                bullish_signals += weight * 0.4
                signal_details.append(f"{timeframe} Price below lower Bollinger Band")
            elif result['bb_signal'] == 'bearish':
                bearish_signals += weight * 0.4
                signal_details.append(f"{timeframe} Price above upper Bollinger Band")
        
        # Calculate total signal strength
        total_signal_strength = bullish_signals + bearish_signals + neutral_signals
        if total_signal_strength == 0:
            total_signal_strength = 1  # Avoid division by zero
        
        # Determine overall sentiment
        overall_sentiment = 'neutral'
        if bullish_signals > bearish_signals and bullish_signals / total_signal_strength > 0.6:
            overall_sentiment = 'bullish'
        elif bearish_signals > bullish_signals and bearish_signals / total_signal_strength > 0.6:
            overall_sentiment = 'bearish'
        
        # Calculate confidence score (0-1)
        confidence = max(bullish_signals, bearish_signals) / total_signal_strength
        
        # Compile final results
        return {
            'symbol': symbol,
            'timeframes': timeframe_results,
            'combined_signals': {
                'bullish': bullish_signals,
                'bearish': bearish_signals,
                'neutral': neutral_signals,
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'signal_details': signal_details
            }
        }
