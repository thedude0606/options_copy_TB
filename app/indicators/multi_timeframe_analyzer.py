"""
Enhanced multi-timeframe analyzer module for options recommendation platform.
Implements dynamic timeframe weighting, adaptive lookback periods, and timeframe confluence analysis.
"""
import pandas as pd
import numpy as np
from app.indicators.patterns.candlestick_patterns import CandlestickPatterns
from app.indicators.technical_indicators import TechnicalIndicators
from datetime import datetime, timedelta

class MultiTimeframeAnalyzer:
    """
    Class to analyze market data across multiple timeframes with enhanced features:
    - Dynamic timeframe weighting based on market conditions
    - Adaptive lookback periods for different market regimes
    - Timeframe confluence analysis to identify stronger signals
    """
    
    def __init__(self, data_collector):
        """
        Initialize the multi-timeframe analyzer
        
        Args:
            data_collector: DataCollector instance for retrieving market data
        """
        self.data_collector = data_collector
        self.technical_indicators = TechnicalIndicators()
        
        # Expanded timeframes list for more comprehensive analysis
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        # Base timeframe weights (will be dynamically adjusted)
        self.base_timeframe_weights = {
            '1m': 0.05,  # Reduced weight for very short timeframe
            '5m': 0.10,
            '15m': 0.15,
            '1h': 0.20,
            '4h': 0.25,
            '1d': 0.25   # Added daily timeframe with significant weight
        }
        
        # Current dynamic weights (initialized to base weights)
        self.timeframe_weights = self.base_timeframe_weights.copy()
        
        # Market regime tracking
        self.current_volatility_regime = 'normal'
        self.trend_strength = 'moderate'
        
        # Confluence tracking
        self.confluence_thresholds = {
            'strong': 0.8,   # 80% agreement across timeframes
            'moderate': 0.6, # 60% agreement across timeframes
            'weak': 0.4      # 40% agreement across timeframes
        }
    
    def fetch_multi_timeframe_data(self, symbol, lookback_days=30):
        """
        Fetch historical data for multiple timeframes with adaptive lookback periods
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Base number of days to look back for historical data
            
        Returns:
            dict: Dictionary with timeframes as keys and DataFrame as values
        """
        multi_timeframe_data = {}
        
        # Determine adaptive lookback based on market regime
        adaptive_lookback = self._calculate_adaptive_lookback(lookback_days)
        
        for timeframe in self.timeframes:
            # Convert timeframe to appropriate parameters for data collector
            if timeframe.endswith('m'):
                frequency_type = 'minute'
                frequency = int(timeframe[:-1])
            elif timeframe.endswith('h'):
                frequency_type = 'hour'
                frequency = int(timeframe[:-1])
            elif timeframe.endswith('d'):
                frequency_type = 'daily'
                frequency = int(timeframe[:-1])
            else:
                frequency_type = 'daily'
                frequency = 1
            
            # Determine appropriate lookback period based on timeframe and market regime
            period_days = self._get_adaptive_period_for_timeframe(timeframe, adaptive_lookback)
            
            # Fetch data from data collector
            try:
                # Determine appropriate period type based on lookback
                period_type = 'day'
                if period_days > 60:
                    period_type = 'month'
                    period_days = period_days // 30 + 1  # Convert to months
                
                data = self.data_collector.get_historical_data(
                    symbol=symbol,
                    period_type=period_type,
                    period=period_days,
                    frequency_type=frequency_type,
                    frequency=frequency
                )
                
                if not data.empty:
                    multi_timeframe_data[timeframe] = data
            except Exception as e:
                print(f"Error fetching {timeframe} data for {symbol}: {str(e)}")
        
        # Update market regime based on the data
        self._update_market_regime(multi_timeframe_data)
        
        # Update dynamic weights based on the current market regime
        self._update_dynamic_weights()
        
        return multi_timeframe_data
    
    def _calculate_adaptive_lookback(self, base_lookback):
        """
        Calculate adaptive lookback period based on market regime
        
        Args:
            base_lookback (int): Base lookback period in days
            
        Returns:
            dict: Adaptive lookback periods for different analysis needs
        """
        # Adjust lookback based on volatility regime
        volatility_multiplier = 1.0
        if self.current_volatility_regime == 'high':
            volatility_multiplier = 0.7  # Shorter lookback in high volatility
        elif self.current_volatility_regime == 'low':
            volatility_multiplier = 1.5  # Longer lookback in low volatility
        
        # Adjust lookback based on trend strength
        trend_multiplier = 1.0
        if self.trend_strength == 'strong':
            trend_multiplier = 0.8  # Shorter lookback in strong trends
        elif self.trend_strength == 'weak':
            trend_multiplier = 1.2  # Longer lookback in weak trends
        
        # Calculate adaptive lookback periods for different analysis needs
        return {
            'standard': max(5, int(base_lookback * volatility_multiplier * trend_multiplier)),
            'pattern_recognition': max(10, int(20 * volatility_multiplier)),
            'trend_analysis': max(20, int(40 * trend_multiplier)),
            'volatility_analysis': max(30, int(60 * volatility_multiplier))
        }
    
    def _get_adaptive_period_for_timeframe(self, timeframe, adaptive_lookback):
        """
        Get appropriate lookback period for a specific timeframe
        
        Args:
            timeframe (str): Timeframe identifier
            adaptive_lookback (dict): Adaptive lookback periods
            
        Returns:
            int: Appropriate lookback period in days
        """
        standard_lookback = adaptive_lookback['standard']
        
        # Scale lookback based on timeframe
        if timeframe == '1m':
            return min(1, standard_lookback // 30)  # Max 1 day for 1-minute data
        elif timeframe == '5m':
            return min(5, standard_lookback // 6)   # Max 5 days for 5-minute data
        elif timeframe == '15m':
            return min(10, standard_lookback // 3)  # Max 10 days for 15-minute data
        elif timeframe == '1h':
            return min(30, standard_lookback)       # Max 30 days for hourly data
        elif timeframe == '4h':
            return standard_lookback                # Full lookback for 4h data
        else:  # '1d'
            return adaptive_lookback['trend_analysis']  # Longer lookback for daily data
    
    def _update_market_regime(self, multi_timeframe_data):
        """
        Update market regime based on the latest data
        
        Args:
            multi_timeframe_data (dict): Dictionary with timeframes as keys and DataFrame as values
        """
        # Use daily data for regime detection if available
        if '1d' in multi_timeframe_data and not multi_timeframe_data['1d'].empty:
            data = multi_timeframe_data['1d']
        # Otherwise use the highest timeframe available
        elif '4h' in multi_timeframe_data and not multi_timeframe_data['4h'].empty:
            data = multi_timeframe_data['4h']
        else:
            # Not enough data to update regime
            return
        
        # Set data for technical indicators
        self.technical_indicators.data = data
        
        # Detect volatility regime
        try:
            volatility_regime = self.technical_indicators.volatility_regime()
            if not volatility_regime.empty:
                self.current_volatility_regime = volatility_regime.iloc[-1]
        except Exception as e:
            print(f"Error detecting volatility regime: {str(e)}")
        
        # Detect trend strength
        try:
            # Calculate ADX for trend strength
            adx = self.technical_indicators.calculate_adx(data)
            if not isinstance(adx, pd.Series) or adx.empty:
                return
                
            latest_adx = adx.iloc[-1]
            
            if latest_adx > 30:
                self.trend_strength = 'strong'
            elif latest_adx > 20:
                self.trend_strength = 'moderate'
            else:
                self.trend_strength = 'weak'
        except Exception as e:
            print(f"Error detecting trend strength: {str(e)}")
    
    def _update_dynamic_weights(self):
        """
        Update timeframe weights based on current market regime
        """
        # Start with base weights
        self.timeframe_weights = self.base_timeframe_weights.copy()
        
        # Adjust weights based on volatility regime
        if self.current_volatility_regime == 'high':
            # In high volatility, favor shorter timeframes
            for tf in self.timeframe_weights:
                if tf in ['1m', '5m', '15m']:
                    self.timeframe_weights[tf] *= 1.3
                elif tf in ['4h', '1d']:
                    self.timeframe_weights[tf] *= 0.7
        elif self.current_volatility_regime == 'low':
            # In low volatility, favor longer timeframes
            for tf in self.timeframe_weights:
                if tf in ['1m', '5m']:
                    self.timeframe_weights[tf] *= 0.7
                elif tf in ['4h', '1d']:
                    self.timeframe_weights[tf] *= 1.3
        
        # Adjust weights based on trend strength
        if self.trend_strength == 'strong':
            # In strong trends, favor longer timeframes
            for tf in self.timeframe_weights:
                if tf in ['1h', '4h', '1d']:
                    self.timeframe_weights[tf] *= 1.2
                elif tf in ['1m', '5m']:
                    self.timeframe_weights[tf] *= 0.8
        elif self.trend_strength == 'weak':
            # In weak trends, favor shorter timeframes for mean reversion
            for tf in self.timeframe_weights:
                if tf in ['1m', '5m', '15m']:
                    self.timeframe_weights[tf] *= 1.2
                elif tf in ['4h', '1d']:
                    self.timeframe_weights[tf] *= 0.8
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.timeframe_weights.values())
        if weight_sum > 0:
            for tf in self.timeframe_weights:
                self.timeframe_weights[tf] /= weight_sum
    
    def analyze_timeframe(self, data, timeframe):
        """
        Analyze a single timeframe's data with enhanced indicators
        
        Args:
            data (pd.DataFrame): OHLC price data for a specific timeframe
            timeframe (str): Timeframe identifier (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            
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
        
        # Add new Phase 1 indicators
        try:
            cmo = self.technical_indicators.chande_momentum_oscillator()
            stoch_rsi = self.technical_indicators.stochastic_rsi()
            obv = self.technical_indicators.on_balance_volume()
            adl = self.technical_indicators.accumulation_distribution_line()
            ama = self.technical_indicators.adaptive_moving_average()
        except Exception as e:
            print(f"Error calculating new indicators: {str(e)}")
            cmo = pd.Series()
            stoch_rsi = pd.DataFrame()
            obv = pd.Series()
            adl = pd.Series()
            ama = pd.Series()
        
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
        
        # Get latest values for new indicators
        latest_cmo = cmo.iloc[-1] if not cmo.empty else None
        latest_stoch_rsi_k = stoch_rsi['k'].iloc[-1] if not stoch_rsi.empty else None
        latest_stoch_rsi_d = stoch_rsi['d'].iloc[-1] if not stoch_rsi.empty else None
        latest_obv = obv.iloc[-1] if not obv.empty else None
        latest_adl = adl.iloc[-1] if not adl.empty else None
        latest_ama = ama.iloc[-1] if not ama.empty else None
        
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
        
        # New indicator signals
        cmo_signal = 'neutral'
        if latest_cmo is not None:
            if latest_cmo < -50:
                cmo_signal = 'bullish'  # Oversold
            elif latest_cmo > 50:
                cmo_signal = 'bearish'  # Overbought
        
        stoch_rsi_signal = 'neutral'
        if latest_stoch_rsi_k is not None and latest_stoch_rsi_d is not None:
            if latest_stoch_rsi_k < 20 and latest_stoch_rsi_d < 20:
                stoch_rsi_signal = 'bullish'  # Oversold
            elif latest_stoch_rsi_k > 80 and latest_stoch_rsi_d > 80:
                stoch_rsi_signal = 'bearish'  # Overbought
            elif latest_stoch_rsi_k > latest_stoch_rsi_d:
                stoch_rsi_signal = 'bullish'  # Crossover up
            elif latest_stoch_rsi_k < latest_stoch_rsi_d:
                stoch_rsi_signal = 'bearish'  # Crossover down
        
        # Volume-based signals
        volume_signal = 'neutral'
        if not data.empty and len(data) > 20:
            # Check if volume is increasing
            avg_volume = data['volume'].rolling(window=20).mean()
            if not avg_volume.empty:
                latest_volume = data['volume'].iloc[-1]
                latest_avg_volume = avg_volume.iloc[-1]
                
                if latest_volume > 1.5 * latest_avg_volume:
                    # High volume, check price direction
                    price_change = data['close'].iloc[-1] - data['open'].iloc[-1]
                    if price_change > 0:
                        volume_signal = 'bullish'  # High volume up day
                    elif price_change < 0:
                        volume_signal = 'bearish'  # High volume down day
        
        # Trend analysis
        trend_signal = 'neutral'
        if not data.empty and len(data) > 50:
            # Simple trend detection using moving averages
            sma20 = data['close'].rolling(window=20).mean()
            sma50 = data['close'].rolling(window=50).mean()
            
            if not sma20.empty and not sma50.empty:
                if sma20.iloc[-1] > sma50.iloc[-1]:
                    trend_signal = 'bullish'  # Uptrend
                elif sma20.iloc[-1] < sma50.iloc[-1]:
                    trend_signal = 'bearish'  # Downtrend
        
        # Compile results with enhanced indicators and signals
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
            'bb_signal': bb_signal,
            # New indicators
            'cmo': latest_cmo,
            'cmo_signal': cmo_signal,
            'stoch_rsi_k': latest_stoch_rsi_k,
            'stoch_rsi_d': latest_stoch_rsi_d,
            'stoch_rsi_signal': stoch_rsi_signal,
            'obv': latest_obv,
            'adl': latest_adl,
            'ama': latest_ama,
            'volume_signal': volume_signal,
            'trend_signal': trend_signal,
            # Market regime info
            'volatility_regime': self.current_volatility_regime,
            'trend_strength': self.trend_strength
        }
    
    def analyze_multi_timeframe(self, symbol, lookback_days=30):
        """
        Analyze market data across multiple timeframes with enhanced confluence detection
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back for historical data
            
        Returns:
            dict: Dictionary with analysis results for each timeframe and combined signals
        """
        # Fetch data for all timeframes with adaptive lookback
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
                    'signal_details': [],
                    'market_regime': {
                        'volatility': self.current_volatility_regime,
                        'trend_strength': self.trend_strength
                    }
                }
            }
        
        # Analyze each timeframe
        timeframe_results = {}
        for timeframe, data in multi_timeframe_data.items():
            # Ensure data is not empty before analysis
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                continue
            timeframe_results[timeframe] = self.analyze_timeframe(data, timeframe)
        
        # Combine signals across timeframes with enhanced confluence detection
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        signal_details = []
        
        # Track signal counts for confluence analysis
        signal_counts = {
            'rsi': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0},
            'macd': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0},
            'bb': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0},
            'cmo': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0},
            'stoch_rsi': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0},
            'volume': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0},
            'trend': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0},
            'candlestick': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0}
        }
        
        # Process each timeframe's signals
        for timeframe, result in timeframe_results.items():
            if not result:
                continue
                
            weight = result['weight']
            
            # Add candlestick pattern signals
            if result['candlestick_sentiment'] == 'bullish':
                bullish_signals += weight * result['pattern_strength']
                signal_counts['candlestick']['bullish'] += 1
                if result['pattern_strength'] > 0:
                    patterns_str = ', '.join([p.replace('_', ' ').title() for p in result['candlestick_patterns'].keys()])
                    signal_details.append(f"{timeframe} Bullish patterns: {patterns_str}")
            elif result['candlestick_sentiment'] == 'bearish':
                bearish_signals += weight * result['pattern_strength']
                signal_counts['candlestick']['bearish'] += 1
                if result['pattern_strength'] > 0:
                    patterns_str = ', '.join([p.replace('_', ' ').title() for p in result['candlestick_patterns'].keys()])
                    signal_details.append(f"{timeframe} Bearish patterns: {patterns_str}")
            else:
                signal_counts['candlestick']['neutral'] += 1
            
            signal_counts['candlestick']['total'] += 1
            
            # Add RSI signals
            if result['rsi_signal'] == 'bullish':
                bullish_signals += weight * 0.5
                signal_counts['rsi']['bullish'] += 1
                signal_details.append(f"{timeframe} RSI oversold: {result['rsi']:.1f}")
            elif result['rsi_signal'] == 'bearish':
                bearish_signals += weight * 0.5
                signal_counts['rsi']['bearish'] += 1
                signal_details.append(f"{timeframe} RSI overbought: {result['rsi']:.1f}")
            else:
                signal_counts['rsi']['neutral'] += 1
            
            signal_counts['rsi']['total'] += 1
            
            # Add MACD signals
            if result['macd_trend'] == 'bullish':
                bullish_signals += weight * 0.6
                signal_counts['macd']['bullish'] += 1
                signal_details.append(f"{timeframe} MACD bullish crossover")
            elif result['macd_trend'] == 'bearish':
                bearish_signals += weight * 0.6
                signal_counts['macd']['bearish'] += 1
                signal_details.append(f"{timeframe} MACD bearish crossover")
            else:
                signal_counts['macd']['neutral'] += 1
            
            signal_counts['macd']['total'] += 1
            
            # Add Bollinger Bands signals
            if result['bb_signal'] == 'bullish':
                bullish_signals += weight * 0.4
                signal_counts['bb']['bullish'] += 1
                signal_details.append(f"{timeframe} Price below lower Bollinger Band")
            elif result['bb_signal'] == 'bearish':
                bearish_signals += weight * 0.4
                signal_counts['bb']['bearish'] += 1
                signal_details.append(f"{timeframe} Price above upper Bollinger Band")
            else:
                signal_counts['bb']['neutral'] += 1
            
            signal_counts['bb']['total'] += 1
            
            # Add new indicator signals
            
            # CMO signals
            if result.get('cmo_signal') == 'bullish':
                bullish_signals += weight * 0.5
                signal_counts['cmo']['bullish'] += 1
                signal_details.append(f"{timeframe} CMO oversold: {result.get('cmo', 'N/A')}")
            elif result.get('cmo_signal') == 'bearish':
                bearish_signals += weight * 0.5
                signal_counts['cmo']['bearish'] += 1
                signal_details.append(f"{timeframe} CMO overbought: {result.get('cmo', 'N/A')}")
            else:
                signal_counts['cmo']['neutral'] += 1
            
            signal_counts['cmo']['total'] += 1
            
            # Stochastic RSI signals
            if result.get('stoch_rsi_signal') == 'bullish':
                bullish_signals += weight * 0.5
                signal_counts['stoch_rsi']['bullish'] += 1
                signal_details.append(f"{timeframe} Stochastic RSI bullish: K={result.get('stoch_rsi_k', 'N/A')}, D={result.get('stoch_rsi_d', 'N/A')}")
            elif result.get('stoch_rsi_signal') == 'bearish':
                bearish_signals += weight * 0.5
                signal_counts['stoch_rsi']['bearish'] += 1
                signal_details.append(f"{timeframe} Stochastic RSI bearish: K={result.get('stoch_rsi_k', 'N/A')}, D={result.get('stoch_rsi_d', 'N/A')}")
            else:
                signal_counts['stoch_rsi']['neutral'] += 1
            
            signal_counts['stoch_rsi']['total'] += 1
            
            # Volume signals
            if result.get('volume_signal') == 'bullish':
                bullish_signals += weight * 0.4
                signal_counts['volume']['bullish'] += 1
                signal_details.append(f"{timeframe} High volume bullish price action")
            elif result.get('volume_signal') == 'bearish':
                bearish_signals += weight * 0.4
                signal_counts['volume']['bearish'] += 1
                signal_details.append(f"{timeframe} High volume bearish price action")
            else:
                signal_counts['volume']['neutral'] += 1
            
            signal_counts['volume']['total'] += 1
            
            # Trend signals
            if result.get('trend_signal') == 'bullish':
                bullish_signals += weight * 0.7
                signal_counts['trend']['bullish'] += 1
                signal_details.append(f"{timeframe} Bullish trend detected")
            elif result.get('trend_signal') == 'bearish':
                bearish_signals += weight * 0.7
                signal_counts['trend']['bearish'] += 1
                signal_details.append(f"{timeframe} Bearish trend detected")
            else:
                signal_counts['trend']['neutral'] += 1
            
            signal_counts['trend']['total'] += 1
        
        # Calculate total signal strength
        total_signal_strength = bullish_signals + bearish_signals + neutral_signals
        if total_signal_strength == 0:
            total_signal_strength = 1  # Avoid division by zero
        
        # Analyze confluence across timeframes and indicators
        confluence_analysis = self._analyze_signal_confluence(signal_counts)
        
        # Apply confluence bonus to signal strength
        if confluence_analysis['bullish_confluence'] >= self.confluence_thresholds['strong']:
            bullish_signals *= 1.5
            signal_details.append(f"Strong bullish confluence detected across {confluence_analysis['bullish_confluence']*100:.0f}% of indicators")
        elif confluence_analysis['bullish_confluence'] >= self.confluence_thresholds['moderate']:
            bullish_signals *= 1.3
            signal_details.append(f"Moderate bullish confluence detected across {confluence_analysis['bullish_confluence']*100:.0f}% of indicators")
        
        if confluence_analysis['bearish_confluence'] >= self.confluence_thresholds['strong']:
            bearish_signals *= 1.5
            signal_details.append(f"Strong bearish confluence detected across {confluence_analysis['bearish_confluence']*100:.0f}% of indicators")
        elif confluence_analysis['bearish_confluence'] >= self.confluence_thresholds['moderate']:
            bearish_signals *= 1.3
            signal_details.append(f"Moderate bearish confluence detected across {confluence_analysis['bearish_confluence']*100:.0f}% of indicators")
        
        # Recalculate total signal strength after confluence adjustment
        total_signal_strength = bullish_signals + bearish_signals + neutral_signals
        if total_signal_strength == 0:
            total_signal_strength = 1  # Avoid division by zero
        
        # Determine overall sentiment with enhanced thresholds
        overall_sentiment = 'neutral'
        if bullish_signals > bearish_signals and bullish_signals / total_signal_strength > 0.55:
            overall_sentiment = 'bullish'
        elif bearish_signals > bullish_signals and bearish_signals / total_signal_strength > 0.55:
            overall_sentiment = 'bearish'
        
        # Calculate confidence score (0-1) with enhanced formula
        raw_confidence = max(bullish_signals, bearish_signals) / total_signal_strength
        
        # Adjust confidence based on confluence
        max_confluence = max(confluence_analysis['bullish_confluence'], confluence_analysis['bearish_confluence'])
        confidence = raw_confidence * (1 + max_confluence) / 2
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        # Compile final results with enhanced information
        return {
            'symbol': symbol,
            'timeframes': timeframe_results,
            'combined_signals': {
                'bullish': bullish_signals,
                'bearish': bearish_signals,
                'neutral': neutral_signals,
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'signal_details': signal_details,
                'market_regime': {
                    'volatility': self.current_volatility_regime,
                    'trend_strength': self.trend_strength
                },
                'confluence_analysis': confluence_analysis,
                'dynamic_weights': self.timeframe_weights
            }
        }
    
    def _analyze_signal_confluence(self, signal_counts):
        """
        Analyze confluence of signals across different indicators and timeframes
        
        Args:
            signal_counts (dict): Dictionary tracking signal counts by indicator type
            
        Returns:
            dict: Confluence analysis results
        """
        # Calculate confluence percentages for each signal type
        confluence = {
            'bullish_by_indicator': {},
            'bearish_by_indicator': {},
            'neutral_by_indicator': {}
        }
        
        total_indicators = 0
        total_bullish = 0
        total_bearish = 0
        total_neutral = 0
        
        for indicator, counts in signal_counts.items():
            if counts['total'] > 0:
                confluence['bullish_by_indicator'][indicator] = counts['bullish'] / counts['total']
                confluence['bearish_by_indicator'][indicator] = counts['bearish'] / counts['total']
                confluence['neutral_by_indicator'][indicator] = counts['neutral'] / counts['total']
                
                total_indicators += 1
                total_bullish += counts['bullish']
                total_bearish += counts['bearish']
                total_neutral += counts['neutral']
        
        # Calculate overall confluence
        total_signals = total_bullish + total_bearish + total_neutral
        
        if total_signals > 0:
            confluence['bullish_confluence'] = total_bullish / total_signals
            confluence['bearish_confluence'] = total_bearish / total_signals
            confluence['neutral_confluence'] = total_neutral / total_signals
        else:
            confluence['bullish_confluence'] = 0
            confluence['bearish_confluence'] = 0
            confluence['neutral_confluence'] = 0
        
        # Find indicators with strongest confluence
        strongest_bullish = max(confluence['bullish_by_indicator'].items(), key=lambda x: x[1], default=('none', 0))
        strongest_bearish = max(confluence['bearish_by_indicator'].items(), key=lambda x: x[1], default=('none', 0))
        
        confluence['strongest_bullish_indicator'] = strongest_bullish[0]
        confluence['strongest_bullish_percentage'] = strongest_bullish[1]
        confluence['strongest_bearish_indicator'] = strongest_bearish[0]
        confluence['strongest_bearish_percentage'] = strongest_bearish[1]
        
        return confluence
