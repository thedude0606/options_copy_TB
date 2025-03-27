"""
Enhanced recommendation engine module for options recommendation platform.
Implements advanced recommendation logic based on technical indicators, options analysis,
and market conditions with improved data generation and caching.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import pickle
import json
from app.indicators.technical_indicators import TechnicalIndicators
from app.analysis.options_analysis import OptionsAnalysis

class RecommendationEngine:
    """
    Enhanced class to generate options trading recommendations based on technical indicators,
    options analysis, and market conditions with improved data generation and caching.
    """
    
    def __init__(self, data_collector):
        """
        Initialize the recommendation engine
        
        Args:
            data_collector: DataCollector instance for retrieving market data
        """
        self.data_collector = data_collector
        self.technical_indicators = TechnicalIndicators()
        self.options_analysis = OptionsAnalysis()
        # Enable debug mode
        self.debug = True
        # Cache settings
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        self.cache_expiry = 15 * 60  # 15 minutes in seconds
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # Track last symbol to detect changes
        self.last_symbol = None
    
    def generate_recommendations(self, symbol, lookback_days=30, confidence_threshold=0.6, strategy_types=None):
        """
        Generate options trading recommendations for a symbol with enhanced filtering and scoring
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back for historical data
            confidence_threshold (float): Minimum confidence score for recommendations
            strategy_types (list): Optional list of strategy types to consider ('directional', 'income', 'volatility')
            
        Returns:
            pd.DataFrame: Recommendations with details
        """
        try:
            # Check if symbol has changed - force cache refresh if needed
            force_refresh = False
            if symbol != self.last_symbol:
                if self.debug:
                    print(f"\n=== SYMBOL CHANGE DETECTED ===")
                    print(f"Symbol changed from {self.last_symbol} to {symbol}, forcing cache refresh")
                force_refresh = True
                self.last_symbol = symbol
            
            if self.debug:
                print(f"\n=== RECOMMENDATION ENGINE DEBUG ===")
                print(f"Generating recommendations for {symbol}")
                print(f"Lookback days: {lookback_days}")
                print(f"Confidence threshold: {confidence_threshold}")
                print(f"Strategy types: {strategy_types if strategy_types else 'All'}")
            
            # Check cache for historical data
            historical_data = self._get_cached_data(f"{symbol}_historical", self._fetch_historical_data, 
                                                   symbol=symbol, lookback_days=lookback_days, force_refresh=force_refresh)
            
            if self.debug:
                print(f"Historical data shape: {historical_data.shape if not historical_data.empty else 'Empty'}")
            
            if historical_data.empty:
                print(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Check cache for options data
            options_data = self._get_cached_data(f"{symbol}_options", self._fetch_options_data, symbol=symbol, force_refresh=force_refresh)
            
            if self.debug:
                print(f"Options data shape: {options_data.shape if not options_data.empty else 'Empty'}")
                if not options_data.empty:
                    print(f"Options data columns: {options_data.columns.tolist()}")
                    print(f"Sample option data (first row):")
                    print(options_data.iloc[0])
            
            if options_data.empty:
                print(f"No options data available for {symbol}")
                return pd.DataFrame()
            
            # Get market context data
            market_context = self._get_market_context(symbol)
            
            if self.debug:
                print(f"Market context: {market_context}")
            
            # Calculate technical indicators with enhanced set
            if self.debug:
                print(f"Calculating technical indicators...")
            
            indicators = self._calculate_indicators(historical_data)
            
            if self.debug:
                print(f"Indicators calculated: {list(indicators.keys())}")
            
            # Calculate options Greeks and probabilities
            if self.debug:
                print(f"Analyzing options data...")
            
            options_analysis = self._analyze_options(options_data)
            
            if self.debug:
                print(f"Options analysis shape: {options_analysis.shape if not options_analysis.empty else 'Empty'}")
                if not options_analysis.empty:
                    print(f"Options analysis columns: {options_analysis.columns.tolist()}")
            
            # Generate signals based on technical indicators and market context
            if self.debug:
                print(f"Generating signals from indicators and market context...")
            
            signals = self._generate_signals(indicators, market_context)
            
            if self.debug:
                print(f"Signal summary: Bullish={signals['bullish']}, Bearish={signals['bearish']}, Neutral={signals['neutral']}")
                print(f"Signal details: {signals['signal_details']}")
            
            # Score options based on signals, options analysis, and strategy preferences
            if self.debug:
                print(f"Scoring options based on signals, analysis, and strategies...")
            
            recommendations = self._score_options(options_analysis, signals, confidence_threshold, strategy_types)
            
            if self.debug:
                print(f"Recommendations shape: {recommendations.shape if not recommendations.empty else 'Empty'}")
                if not recommendations.empty:
                    print(f"Recommendations columns: {recommendations.columns.tolist()}")
                    print(f"Top recommendation:")
                    print(recommendations.iloc[0])
                else:
                    print(f"No recommendations generated that meet the confidence threshold")
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    
    def _get_cached_data(self, cache_key, fetch_function, **kwargs):
        """
        Get data from cache if available and not expired, otherwise fetch and cache
        
        Args:
            cache_key (str): Key for the cached data
            fetch_function (callable): Function to call if cache miss
            **kwargs: Arguments to pass to fetch_function
                force_refresh (bool): If True, ignore cache and fetch fresh data
            
        Returns:
            pd.DataFrame: The requested data
        """
        # Extract force_refresh parameter if present
        force_refresh = kwargs.pop('force_refresh', False)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Check if cache file exists and is not expired, unless force_refresh is True
        if not force_refresh and os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < self.cache_expiry:
                if self.debug:
                    print(f"Loading {cache_key} from cache (age: {file_age:.1f}s)")
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading cache: {str(e)}")
                    # Continue to fetch data if cache loading fails
            else:
                if self.debug:
                    print(f"Cache expired for {cache_key} (age: {file_age:.1f}s)")
        else:
            if force_refresh:
                if self.debug:
                    print(f"=== FORCE REFRESH ACTIVE ===")
                    print(f"Force refresh requested for {cache_key}, fetching fresh data")
            else:
                if self.debug:
                    print(f"No cache found for {cache_key}")
        
        # Fetch data
        data = fetch_function(**kwargs)
        
        # Cache data if not empty
        if not data.empty:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                if self.debug:
                    print(f"Cached {cache_key} data")
            except Exception as e:
                print(f"Error caching data: {str(e)}")
        
        return data
    
    def _fetch_historical_data(self, symbol, lookback_days=30):
        """
        Fetch historical data with optimized parameters
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Historical price data
        """
        # Determine optimal period type and frequency based on lookback days
        if lookback_days <= 5:
            period_type = 'day'
            period = 5
            frequency_type = 'minute'
            frequency = 30
        elif lookback_days <= 10:
            period_type = 'day'
            period = 10
            frequency_type = 'minute'
            frequency = 60
        elif lookback_days <= 30:
            period_type = 'month'
            period = 1
            frequency_type = 'daily'
            frequency = 1
        else:
            period_type = 'year'
            period = 1
            frequency_type = 'daily'
            frequency = 1
        
        if self.debug:
            print(f"Fetching historical data with period_type={period_type}, period={period}, frequency_type={frequency_type}, frequency={frequency}")
        
        return self.data_collector.get_historical_data(
            symbol=symbol,
            period_type=period_type,
            period=period,
            frequency_type=frequency_type,
            frequency=frequency
        )
    
    def _fetch_options_data(self, symbol):
        """
        Fetch options data with optimized parameters
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            pd.DataFrame: Options data
        """
        if self.debug:
            print(f"Fetching options data for {symbol}")
        
        return self.data_collector.get_option_data(symbol)
    
    def _get_market_context(self, symbol):
        """
        Get market context data including volatility index, sector performance, and earnings dates
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Market context data
        """
        context = {
            'market_volatility': 'normal',
            'sector_trend': 'neutral',
            'earnings_proximity': 'far',
            'overall_market_trend': 'neutral'
        }
        
        try:
            # Get VIX data for market volatility
            vix_data = self.data_collector.get_historical_data(
                symbol='VIX',
                period_type='month',
                period=1,
                frequency_type='daily',
                frequency=1
            )
            
            if not vix_data.empty:
                latest_vix = vix_data['close'].iloc[-1]
                if latest_vix > 25:
                    context['market_volatility'] = 'high'
                elif latest_vix < 15:
                    context['market_volatility'] = 'low'
                
                # Calculate 10-day VIX trend
                if len(vix_data) >= 10:
                    vix_10d_avg = vix_data['close'].iloc[-10:].mean()
                    vix_trend = (latest_vix / vix_10d_avg) - 1
                    if vix_trend > 0.1:
                        context['volatility_trend'] = 'rising'
                    elif vix_trend < -0.1:
                        context['volatility_trend'] = 'falling'
                    else:
                        context['volatility_trend'] = 'stable'
            
            # Get SPY data for overall market trend
            spy_data = self.data_collector.get_historical_data(
                symbol='SPY',
                period_type='month',
                period=1,
                frequency_type='daily',
                frequency=1
            )
            
            if not spy_data.empty:
                # Calculate 10-day SPY trend
                if len(spy_data) >= 10:
                    spy_latest = spy_data['close'].iloc[-1]
                    spy_10d_avg = spy_data['close'].iloc[-10:].mean()
                    spy_trend = (spy_latest / spy_10d_avg) - 1
                    if spy_trend > 0.03:
                        context['overall_market_trend'] = 'bullish'
                    elif spy_trend < -0.03:
                        context['overall_market_trend'] = 'bearish'
            
            # Try to get sector ETF data based on symbol's sector
            # This is a simplified approach - in a real implementation, you would have a mapping of symbols to sectors
            sector_etfs = {
                'XLF': 'Financial',
                'XLK': 'Technology',
                'XLE': 'Energy',
                'XLV': 'Healthcare',
                'XLI': 'Industrial',
                'XLP': 'Consumer Staples',
                'XLY': 'Consumer Discretionary',
                'XLB': 'Materials',
                'XLU': 'Utilities',
                'XLRE': 'Real Estate'
            }
            
            # For demonstration, we'll use SPY as a proxy for sector trend
            # In a real implementation, you would determine the symbol's sector and use the appropriate ETF
            if not spy_data.empty:
                latest_close = spy_data['close'].iloc[-1]
                prev_close = spy_data['close'].iloc[-2] if len(spy_data) > 1 else latest_close
                sector_change = (latest_close / prev_close) - 1
                
                if sector_change > 0.01:
                    context['sector_trend'] = 'bullish'
                elif sector_change < -0.01:
                    context['sector_trend'] = 'bearish'
            
            # Check for upcoming earnings (simplified)
            # In a real implementation, you would query an earnings calendar API
            context['earnings_proximity'] = 'far'  # Default assumption
            
        except Exception as e:
            print(f"Error getting market context: {str(e)}")
        
        return context
    
    def _calculate_indicators(self, historical_data):
        """
        Calculate enhanced set of technical indicators for historical data
        
        Args:
            historical_data (pd.DataFrame): Historical price data
            
        Returns:
            dict: Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate RSI
        indicators['rsi'] = self.technical_indicators.calculate_rsi(historical_data)
        
        # Calculate MACD
        macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(historical_data)
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram
        
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = self.technical_indicators.calculate_bollinger_bands(historical_data)
        indicators['bollinger_middle'] = middle_band
        indicators['bollinger_upper'] = upper_band
        indicators['bollinger_lower'] = lower_band
        
        # Calculate IMI
        indicators['imi'] = self.technical_indicators.calculate_imi(historical_data)
        
        # Calculate MFI
        indicators['mfi'] = self.technical_indicators.calculate_mfi(historical_data)
        
        # Calculate Fair Value Gap
        indicators['fvg'] = self.technical_indicators.calculate_fair_value_gap(historical_data)
        
        # Calculate Liquidity Zones
        indicators['liquidity_zones'] = self.technical_indicators.calculate_liquidity_zones(historical_data)
        
        # Calculate Moving Averages
        indicators['moving_averages'] = self.technical_indicators.calculate_moving_averages(historical_data)
        
        # Calculate Volatility
        indicators['volatility'] = self.technical_indicators.calculate_volatility(historical_data)
        
        # Calculate additional indicators
        
        # ATR (Average True Range) for volatility measurement
        try:
            high = historical_data['high']
            low = historical_data['low']
            close = historical_data['close']
            
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            indicators['atr'] = atr
            
            # ATR percentage (ATR relative to price)
            indicators['atr_percentage'] = atr / close * 100
        except Exception as e:
            print(f"Error calculating ATR: {str(e)}")
        
        # Stochastic Oscillator
        try:
            high_14 = historical_data['high'].rolling(window=14).max()
            low_14 = historical_data['low'].rolling(window=14).min()
            
            # Fast Stochastic
            k_fast = 100 * (close - low_14) / (high_14 - low_14)
            # Slow Stochastic
            d_slow = k_fast.rolling(window=3).mean()
            
            indicators['stochastic_k'] = k_fast
            indicators['stochastic_d'] = d_slow
        except Exception as e:
            print(f"Error calculating Stochastic Oscillator: {str(e)}")
        
        # On-Balance Volume (OBV)
        try:
            obv = pd.Series(0, index=close.index)
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + historical_data['volume'].iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - historical_data['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            indicators['obv'] = obv
        except Exception as e:
            print(f"Error calculating OBV: {str(e)}")
        
        # Price Rate of Change (ROC)
        try:
            n = 12  # 12-day ROC
            roc = ((close / close.shift(n)) - 1) * 100
            indicators['roc'] = roc
        except Exception as e:
            print(f"Error calculating ROC: {str(e)}")
        
        # Ichimoku Cloud
        try:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = historical_data['high'].rolling(window=9).max()
            period9_low = historical_data['low'].rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = historical_data['high'].rolling(window=26).max()
            period26_low = historical_data['low'].rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            period52_high = historical_data['high'].rolling(window=52).max()
            period52_low = historical_data['low'].rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted back 26 periods
            chikou_span = close.shift(-26)
            
            indicators['ichimoku_tenkan'] = tenkan_sen
            indicators['ichimoku_kijun'] = kijun_sen
            indicators['ichimoku_senkou_a'] = senkou_span_a
            indicators['ichimoku_senkou_b'] = senkou_span_b
            indicators['ichimoku_chikou'] = chikou_span
        except Exception as e:
            print(f"Error calculating Ichimoku Cloud: {str(e)}")
        
        return indicators
    
    def _analyze_options(self, options_data):
        """
        Analyze options data with enhanced Greeks and probabilities
        
        Args:
            options_data (pd.DataFrame): Options data
            
        Returns:
            pd.DataFrame: Analyzed options data
        """
        if self.debug:
            print(f"Analyzing options with enhanced Greeks calculation...")
        
        # Calculate Greeks
        options_with_greeks = self.options_analysis.calculate_all_greeks(options_data)
        
        if self.debug:
            print(f"Options with Greeks shape: {options_with_greeks.shape if not options_with_greeks.empty else 'Empty'}")
            if not options_with_greeks.empty:
                print(f"Greek columns: {[col for col in options_with_greeks.columns if col in ['delta', 'gamma', 'theta', 'vega', 'rho']]}")
        
        # Calculate probability of profit
        if self.debug:
            print(f"Calculating probability of profit...")
        
        options_with_prob = self.options_analysis.calculate_probability_of_profit(options_with_greeks)
        
        if self.debug:
            if 'probabilityOfProfit' in options_with_prob.columns:
                print(f"Probability of profit stats: min={options_with_prob['probabilityOfProfit'].min()}, max={options_with_prob['probabilityOfProfit'].max()}, mean={options_with_prob['probabilityOfProfit'].mean()}")
            else:
                print(f"Warning: probabilityOfProfit column not found after calculation")
        
        # Calculate risk-reward ratio
        if self.debug:
            print(f"Calculating risk-reward ratio...")
        
        analyzed_options = self.options_analysis.calculate_risk_reward_ratio(options_with_prob)
        
        if self.debug:
            if 'riskRewardRatio' in analyzed_options.columns:
                print(f"Risk-reward ratio stats: min={analyzed_options['riskRewardRatio'].min()}, max={analyzed_options['riskRewardRatio'].max()}, mean={analyzed_options['riskRewardRatio'].mean()}")
            else:
                print(f"Warning: riskRewardRatio column not found after calculation")
        
        # Calculate additional metrics
        
        # Implied Volatility Rank (IVR)
        try:
            # Group by expiration date and calculate average IV
            iv_by_expiration = analyzed_options.groupby('expirationDate')['impliedVolatility'].mean().reset_index()
            
            # Calculate IV percentile (simplified - in a real implementation, you would compare to historical IV)
            # For demonstration, we'll use a random value between 0 and 100
            iv_rank = np.random.randint(0, 100)
            
            # Add IV rank to all options
            analyzed_options['ivRank'] = iv_rank
            
            if self.debug:
                print(f"Added IV Rank: {iv_rank}")
        except Exception as e:
            print(f"Error calculating IV Rank: {str(e)}")
        
        # Calculate expected move based on IV
        try:
            # Get the underlying price
            if 'underlyingPrice' in analyzed_options.columns:
                underlying_price = analyzed_options['underlyingPrice'].iloc[0]
                
                # Calculate expected move for each option based on its IV and days to expiration
                analyzed_options['expectedMove'] = underlying_price * analyzed_options['impliedVolatility'] * np.sqrt(analyzed_options['daysToExpiration'] / 365)
                
                if self.debug:
                    print(f"Added expected move based on IV and DTE")
            else:
                print(f"Warning: underlyingPrice not found, cannot calculate expected move")
        except Exception as e:
            print(f"Error calculating expected move: {str(e)}")
        
        # Calculate option liquidity score
        try:
            # Normalize volume and open interest to 0-1 scale
            max_volume = analyzed_options['volume'].max() if 'volume' in analyzed_options.columns else 1
            max_oi = analyzed_options['openInterest'].max() if 'openInterest' in analyzed_options.columns else 1
            
            volume_score = analyzed_options['volume'] / max_volume if max_volume > 0 else 0
            oi_score = analyzed_options['openInterest'] / max_oi if max_oi > 0 else 0
            
            # Calculate bid-ask spread percentage
            if 'bid' in analyzed_options.columns and 'ask' in analyzed_options.columns:
                spread_pct = (analyzed_options['ask'] - analyzed_options['bid']) / ((analyzed_options['ask'] + analyzed_options['bid']) / 2)
                # Invert spread so lower spread = higher score
                spread_score = 1 - np.clip(spread_pct, 0, 1)
            else:
                spread_score = 0
            
            # Combine into liquidity score (weighted average)
            analyzed_options['liquidityScore'] = (volume_score * 0.3) + (oi_score * 0.3) + (spread_score * 0.4)
            
            if self.debug:
                print(f"Added liquidity score based on volume, open interest, and bid-ask spread")
        except Exception as e:
            print(f"Error calculating liquidity score: {str(e)}")
        
        return analyzed_options
    
    def _generate_signals(self, indicators, market_context):
        """
        Generate enhanced trading signals based on technical indicators and market context
        
        Args:
            indicators (dict): Dictionary of calculated indicators
            market_context (dict): Market context data
            
        Returns:
            dict: Dictionary of trading signals
        """
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        signal_details = []
        
        # RSI signals
        if 'rsi' in indicators and not indicators['rsi'].empty:
            current_rsi = indicators['rsi'].iloc[-1]
            
            if current_rsi < 30:
                bullish_signals += 1
                signal_details.append(f"RSI oversold ({current_rsi:.1f})")
            elif current_rsi > 70:
                bearish_signals += 1
                signal_details.append(f"RSI overbought ({current_rsi:.1f})")
            else:
                neutral_signals += 1
                signal_details.append(f"RSI neutral ({current_rsi:.1f})")
        
        # MACD signals
        if all(k in indicators for k in ['macd_line', 'macd_signal', 'macd_histogram']):
            if not indicators['macd_line'].empty and not indicators['macd_signal'].empty:
                current_macd = indicators['macd_line'].iloc[-1]
                current_signal = indicators['macd_signal'].iloc[-1]
                current_hist = indicators['macd_histogram'].iloc[-1]
                prev_hist = indicators['macd_histogram'].iloc[-2] if len(indicators['macd_histogram']) > 1 else 0
                
                # MACD line crosses above signal line
                if current_macd > current_signal and current_hist > 0 and prev_hist < 0:
                    bullish_signals += 1
                    signal_details.append("MACD bullish crossover")
                # MACD line crosses below signal line
                elif current_macd < current_signal and current_hist < 0 and prev_hist > 0:
                    bearish_signals += 1
                    signal_details.append("MACD bearish crossover")
                # MACD above zero line
                elif current_macd > 0:
                    bullish_signals += 0.5
                    signal_details.append("MACD above zero")
                # MACD below zero line
                elif current_macd < 0:
                    bearish_signals += 0.5
                    signal_details.append("MACD below zero")
                else:
                    neutral_signals += 1
                    signal_details.append("MACD neutral")
        
        # Bollinger Bands signals
        if all(k in indicators for k in ['bollinger_middle', 'bollinger_upper', 'bollinger_lower']):
            if not indicators['bollinger_middle'].empty:
                close = indicators['bollinger_middle'].iloc[-1]  # Using middle band as a proxy for close
                upper = indicators['bollinger_upper'].iloc[-1]
                lower = indicators['bollinger_lower'].iloc[-1]
                
                # Price near upper band
                if close > upper * 0.95:
                    bearish_signals += 0.5
                    signal_details.append("Price near upper Bollinger Band")
                # Price near lower band
                elif close < lower * 1.05:
                    bullish_signals += 0.5
                    signal_details.append("Price near lower Bollinger Band")
                # Price in middle of bands
                else:
                    neutral_signals += 1
                    signal_details.append("Price within Bollinger Bands")
        
        # Stochastic Oscillator signals
        if all(k in indicators for k in ['stochastic_k', 'stochastic_d']):
            if not indicators['stochastic_k'].empty and not indicators['stochastic_d'].empty:
                k = indicators['stochastic_k'].iloc[-1]
                d = indicators['stochastic_d'].iloc[-1]
                
                # Oversold
                if k < 20 and d < 20:
                    bullish_signals += 1
                    signal_details.append(f"Stochastic oversold (K={k:.1f}, D={d:.1f})")
                # Overbought
                elif k > 80 and d > 80:
                    bearish_signals += 1
                    signal_details.append(f"Stochastic overbought (K={k:.1f}, D={d:.1f})")
                # K crosses above D
                elif k > d and indicators['stochastic_k'].iloc[-2] < indicators['stochastic_d'].iloc[-2]:
                    bullish_signals += 0.5
                    signal_details.append("Stochastic K crosses above D")
                # K crosses below D
                elif k < d and indicators['stochastic_k'].iloc[-2] > indicators['stochastic_d'].iloc[-2]:
                    bearish_signals += 0.5
                    signal_details.append("Stochastic K crosses below D")
                else:
                    neutral_signals += 0.5
                    signal_details.append("Stochastic neutral")
        
        # Ichimoku Cloud signals
        if all(k in indicators for k in ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b']):
            if not indicators['ichimoku_tenkan'].empty and not indicators['ichimoku_kijun'].empty:
                tenkan = indicators['ichimoku_tenkan'].iloc[-1]
                kijun = indicators['ichimoku_kijun'].iloc[-1]
                senkou_a = indicators['ichimoku_senkou_a'].iloc[-1]
                senkou_b = indicators['ichimoku_senkou_b'].iloc[-1]
                
                # Price above the cloud
                if tenkan > senkou_a and tenkan > senkou_b:
                    bullish_signals += 1
                    signal_details.append("Price above Ichimoku Cloud")
                # Price below the cloud
                elif tenkan < senkou_a and tenkan < senkou_b:
                    bearish_signals += 1
                    signal_details.append("Price below Ichimoku Cloud")
                # Tenkan crosses above Kijun
                elif tenkan > kijun and indicators['ichimoku_tenkan'].iloc[-2] < indicators['ichimoku_kijun'].iloc[-2]:
                    bullish_signals += 0.5
                    signal_details.append("Tenkan crosses above Kijun")
                # Tenkan crosses below Kijun
                elif tenkan < kijun and indicators['ichimoku_tenkan'].iloc[-2] > indicators['ichimoku_kijun'].iloc[-2]:
                    bearish_signals += 0.5
                    signal_details.append("Tenkan crosses below Kijun")
                else:
                    neutral_signals += 0.5
                    signal_details.append("Ichimoku neutral")
        
        # Market context signals
        if market_context:
            # Market volatility
            if market_context.get('market_volatility') == 'high':
                signal_details.append("High market volatility")
            elif market_context.get('market_volatility') == 'low':
                signal_details.append("Low market volatility")
            
            # Sector trend
            if market_context.get('sector_trend') == 'bullish':
                bullish_signals += 0.5
                signal_details.append("Bullish sector trend")
            elif market_context.get('sector_trend') == 'bearish':
                bearish_signals += 0.5
                signal_details.append("Bearish sector trend")
            
            # Overall market trend
            if market_context.get('overall_market_trend') == 'bullish':
                bullish_signals += 0.5
                signal_details.append("Bullish overall market")
            elif market_context.get('overall_market_trend') == 'bearish':
                bearish_signals += 0.5
                signal_details.append("Bearish overall market")
        
        return {
            'bullish': bullish_signals,
            'bearish': bearish_signals,
            'neutral': neutral_signals,
            'signal_details': signal_details
        }
    
    def _score_options(self, options_data, signals, confidence_threshold, strategy_types=None):
        """
        Score options based on signals, options analysis, and strategy preferences with enhanced algorithm
        
        Args:
            options_data (pd.DataFrame): Analyzed options data
            signals (dict): Dictionary of trading signals
            confidence_threshold (float): Minimum confidence score for recommendations
            strategy_types (list): Optional list of strategy types to consider
            
        Returns:
            pd.DataFrame: Scored options recommendations
        """
        if self.debug:
            print(f"Scoring options with confidence threshold: {confidence_threshold}")
            print(f"Options data shape before scoring: {options_data.shape if not options_data.empty else 'Empty'}")
            if not options_data.empty:
                print(f"Options data columns: {options_data.columns.tolist()}")
                print(f"First option before scoring:")
                print(options_data.iloc[0])
        
        if options_data.empty:
            if self.debug:
                print(f"Error: Options data is empty, cannot score")
            return pd.DataFrame()
        
        # Determine overall market direction
        bullish_signals = signals['bullish']
        bearish_signals = signals['bearish']
        neutral_signals = signals['neutral']
        
        total_signals = bullish_signals + bearish_signals + neutral_signals
        if total_signals == 0:
            if self.debug:
                print(f"Error: No signals available (total_signals=0)")
            return pd.DataFrame()
        
        bullish_score = bullish_signals / total_signals
        bearish_score = bearish_signals / total_signals
        neutral_score = neutral_signals / total_signals
        
        if self.debug:
            print(f"Market direction scores: Bullish={bullish_score:.2f}, Bearish={bearish_score:.2f}, Neutral={neutral_score:.2f}")
        
        market_direction = 'neutral'
        if bullish_score > 0.5 and bullish_score > bearish_score:
            market_direction = 'bullish'
        elif bearish_score > 0.5 and bearish_score > bullish_score:
            market_direction = 'bearish'
        
        if self.debug:
            print(f"Determined market direction: {market_direction}")
        
        # Define strategy types if not provided
        if strategy_types is None:
            strategy_types = ['directional', 'income', 'volatility']
        
        # Filter options based on market direction and strategy types
        filtered_options = options_data.copy()
        
        # Apply strategy-specific filters
        if 'directional' in strategy_types:
            if market_direction == 'bullish':
                # For bullish market, include calls
                directional_mask = (options_data['optionType'] == 'CALL')
                if self.debug:
                    print(f"Including CALL options for directional bullish strategy")
            elif market_direction == 'bearish':
                # For bearish market, include puts
                directional_mask = (options_data['optionType'] == 'PUT')
                if self.debug:
                    print(f"Including PUT options for directional bearish strategy")
            else:
                # For neutral market, include both
                directional_mask = pd.Series(True, index=options_data.index)
                if self.debug:
                    print(f"Including all options for directional neutral strategy")
        else:
            directional_mask = pd.Series(False, index=options_data.index)
        
        if 'income' in strategy_types:
            # For income strategies, prefer options with high theta and moderate delta
            if 'theta' in options_data.columns and 'delta' in options_data.columns:
                income_mask = (
                    (options_data['theta'] < -0.01) &  # Negative theta (time decay)
                    (options_data['delta'].abs() < 0.7) &  # Not too deep ITM or OTM
                    (options_data['delta'].abs() > 0.2)  # Not too far OTM
                )
                if self.debug:
                    print(f"Including options with high theta and moderate delta for income strategy")
            else:
                income_mask = pd.Series(False, index=options_data.index)
        else:
            income_mask = pd.Series(False, index=options_data.index)
        
        if 'volatility' in strategy_types:
            # For volatility strategies, prefer options with high vega
            if 'vega' in options_data.columns and 'ivRank' in options_data.columns:
                volatility_mask = (
                    (options_data['vega'] > 0.05) &  # High vega
                    (
                        # Low IV rank for long volatility, high IV rank for short volatility
                        (options_data['ivRank'] < 30) |
                        (options_data['ivRank'] > 70)
                    )
                )
                if self.debug:
                    print(f"Including options with high vega and extreme IV rank for volatility strategy")
            else:
                volatility_mask = pd.Series(False, index=options_data.index)
        else:
            volatility_mask = pd.Series(False, index=options_data.index)
        
        # Combine all strategy masks
        combined_mask = directional_mask | income_mask | volatility_mask
        filtered_options = options_data[combined_mask]
        
        if filtered_options.empty:
            if self.debug:
                print(f"Error: No options left after filtering by strategies")
            # Fall back to all options if filtering results in empty set
            filtered_options = options_data
            if self.debug:
                print(f"Falling back to all options")
        
        if self.debug:
            print(f"Options data shape after filtering: {filtered_options.shape}")
        
        # Score each option with enhanced algorithm
        scores = []
        for idx, row in filtered_options.iterrows():
            if self.debug and idx == 0:
                print(f"Scoring first option (index {idx}):")
                print(f"Option: {row['symbol'] if 'symbol' in row else 'Unknown'}, Type: {row['optionType'] if 'optionType' in row else 'Unknown'}, Strike: {row['strikePrice'] if 'strikePrice' in row else 'Unknown'}")
            
            score = 0
            strategy = ""
            
            # Base score from market direction
            if market_direction == 'bullish' and row['optionType'] == 'CALL':
                score += bullish_score * 20  # 20% weight
                strategy = "Directional Bullish"
                if self.debug and idx == 0:
                    print(f"  Added bullish score: +{bullish_score * 20:.2f}")
            elif market_direction == 'bearish' and row['optionType'] == 'PUT':
                score += bearish_score * 20  # 20% weight
                strategy = "Directional Bearish"
                if self.debug and idx == 0:
                    print(f"  Added bearish score: +{bearish_score * 20:.2f}")
            elif market_direction == 'neutral':
                score += neutral_score * 20  # 20% weight
                strategy = "Neutral"
                if self.debug and idx == 0:
                    print(f"  Added neutral score: +{neutral_score * 20:.2f}")
            
            # Score based on probability of profit
            if 'probabilityOfProfit' in row and not pd.isna(row['probabilityOfProfit']):
                pop_score = row['probabilityOfProfit']
                score += pop_score * 20  # 20% weight
                if self.debug and idx == 0:
                    print(f"  Added probability score: +{pop_score * 20:.2f} (POP: {pop_score:.2f})")
            else:
                if self.debug and idx == 0:
                    print(f"  Warning: probabilityOfProfit not available")
            
            # Score based on risk-reward ratio
            if 'riskRewardRatio' in row and not pd.isna(row['riskRewardRatio']):
                rr_ratio = row['riskRewardRatio']
                if rr_ratio > 0:
                    rr_score = min(rr_ratio / 3, 1)  # Cap at 1
                    score += rr_score * 15  # 15% weight
                    if self.debug and idx == 0:
                        print(f"  Added risk-reward score: +{rr_score * 15:.2f} (RR: {rr_ratio:.2f})")
            else:
                if self.debug and idx == 0:
                    print(f"  Warning: riskRewardRatio not available")
            
            # Score based on delta (prefer 0.3-0.7 range)
            if 'delta' in row and not pd.isna(row['delta']):
                delta = abs(row['delta'])
                if 0.3 <= delta <= 0.7:
                    delta_score = 1 - abs(delta - 0.5) / 0.5
                else:
                    delta_score = 0.2  # Lower score for very low or high delta
                
                score += delta_score * 15  # 15% weight
                if self.debug and idx == 0:
                    print(f"  Added delta score: +{delta_score * 15:.2f} (Delta: {delta:.2f})")
            else:
                if self.debug and idx == 0:
                    print(f"  Warning: delta not available")
            
            # Score based on days to expiration
            if 'daysToExpiration' in row and not pd.isna(row['daysToExpiration']):
                # Prefer 30-45 DTE
                # Convert Timestamp to days if it's a Timestamp object
                if isinstance(row['daysToExpiration'], pd.Timedelta):
                    days = row['daysToExpiration'].days
                else:
                    # If it's already a number, use it directly
                    try:
                        days = float(row['daysToExpiration'])
                    except (ValueError, TypeError):
                        days = 0
                        if self.debug and idx == 0:
                            print(f"  Warning: Could not convert daysToExpiration to float: {row['daysToExpiration']}")
                
                if 20 <= days <= 60:
                    dte_score = 1 - abs(days - 40) / 40
                else:
                    dte_score = 0.2  # Lower score for very short or long DTE
                
                score += dte_score * 10  # 10% weight
                if self.debug and idx == 0:
                    print(f"  Added DTE score: +{dte_score * 10:.2f} (Days: {days})")
            else:
                if self.debug and idx == 0:
                    print(f"  Warning: daysToExpiration not available")
            
            # Score based on liquidity
            if 'liquidityScore' in row and not pd.isna(row['liquidityScore']):
                liquidity_score = row['liquidityScore']
                score += liquidity_score * 10  # 10% weight
                if self.debug and idx == 0:
                    print(f"  Added liquidity score: +{liquidity_score * 10:.2f}")
            else:
                # Fallback liquidity scoring based on volume and open interest
                liquidity_score = 0
                if 'volume' in row and not pd.isna(row['volume']) and row['volume'] > 100:
                    liquidity_score += 0.5
                if 'openInterest' in row and not pd.isna(row['openInterest']) and row['openInterest'] > 500:
                    liquidity_score += 0.5
                
                score += liquidity_score * 10  # 10% weight
                if self.debug and idx == 0:
                    print(f"  Added fallback liquidity score: +{liquidity_score * 10:.2f}")
            
            # Score based on bid-ask spread
            if 'bid' in row and 'ask' in row and not pd.isna(row['bid']) and not pd.isna(row['ask']):
                mid_price = (row['bid'] + row['ask']) / 2
                if mid_price > 0:
                    spread_pct = (row['ask'] - row['bid']) / mid_price
                    # Lower spread is better
                    spread_score = max(0, 1 - (spread_pct * 10))  # Penalize spreads > 10%
                    score += spread_score * 10  # 10% weight
                    if self.debug and idx == 0:
                        print(f"  Added spread score: +{spread_score * 10:.2f} (Spread: {spread_pct:.2%})")
            
            if self.debug and idx == 0:
                print(f"  Final score: {score:.2f}, Confidence: {score / 100:.2f}")
            
            # Determine strategy type based on option characteristics
            if not strategy:
                if 'theta' in row and not pd.isna(row['theta']) and row['theta'] < -0.01:
                    if 'delta' in row and not pd.isna(row['delta']):
                        delta = abs(row['delta'])
                        if delta > 0.5:
                            strategy = "Income (High Delta)"
                        else:
                            strategy = "Income (Low Delta)"
                elif 'vega' in row and not pd.isna(row['vega']) and row['vega'] > 0.05:
                    if 'ivRank' in row and not pd.isna(row['ivRank']):
                        if row['ivRank'] < 30:
                            strategy = "Long Volatility"
                        elif row['ivRank'] > 70:
                            strategy = "Short Volatility"
                        else:
                            strategy = "Volatility Neutral"
                    else:
                        strategy = "Volatility"
                else:
                    strategy = "Balanced"
            
            # Get underlying price
            underlying_price = 0
            if 'underlyingPrice' in row and not pd.isna(row['underlyingPrice']):
                underlying_price = row['underlyingPrice']
            
            # Calculate entry price (mid price)
            entry_price = 0
            if 'bid' in row and 'ask' in row and not pd.isna(row['bid']) and not pd.isna(row['ask']):
                entry_price = (row['bid'] + row['ask']) / 2
            
            # Add to scores list
            scores.append({
                'symbol': row['symbol'] if 'symbol' in row else '',
                'optionType': row['optionType'] if 'optionType' in row else 'UNKNOWN',
                'strikePrice': row['strikePrice'] if 'strikePrice' in row else 0,
                'expirationDate': row['expirationDate'] if 'expirationDate' in row else 'UNKNOWN',
                'bid': row['bid'] if 'bid' in row else 0,
                'ask': row['ask'] if 'ask' in row else 0,
                'entryPrice': entry_price,
                'underlyingPrice': underlying_price,
                'delta': row['delta'] if 'delta' in row else 0,
                'gamma': row['gamma'] if 'gamma' in row else 0,
                'theta': row['theta'] if 'theta' in row else 0,
                'vega': row['vega'] if 'vega' in row else 0,
                'rho': row['rho'] if 'rho' in row else 0,
                'probabilityOfProfit': row['probabilityOfProfit'] if 'probabilityOfProfit' in row else 0,
                'riskRewardRatio': row['riskRewardRatio'] if 'riskRewardRatio' in row else 0,
                'potentialReturn': row['potentialReturn'] if 'potentialReturn' in row else 0,
                'daysToExpiration': days if 'daysToExpiration' in row and not pd.isna(row['daysToExpiration']) else 0,
                'liquidityScore': row['liquidityScore'] if 'liquidityScore' in row else 0,
                'ivRank': row['ivRank'] if 'ivRank' in row else 0,
                'expectedMove': row['expectedMove'] if 'expectedMove' in row else 0,
                'score': score,
                'confidence': score / 100,  # Convert to 0-1 scale
                'marketDirection': market_direction,
                'strategy': strategy,
                'signalDetails': signals['signal_details']
            })
        
        # Convert to DataFrame and filter by confidence threshold
        recommendations_df = pd.DataFrame(scores)
        
        if self.debug:
            print(f"Created recommendations DataFrame with {len(scores)} rows")
            if not recommendations_df.empty:
                print(f"Recommendations columns: {recommendations_df.columns.tolist()}")
        
        if not recommendations_df.empty:
            # Filter by confidence threshold, but ensure we have at least some recommendations
            filtered_recommendations = recommendations_df[recommendations_df['confidence'] >= confidence_threshold]
        
            if self.debug:
                print(f"Filtered recommendations by confidence >= {confidence_threshold}: {len(filtered_recommendations)} rows remaining")
            
            # If no recommendations meet the threshold, lower it to get at least the top 3
            if filtered_recommendations.empty:
                if self.debug:
                    print(f"No recommendations meet the confidence threshold, using top 3 recommendations instead")
                filtered_recommendations = recommendations_df.nlargest(3, 'confidence')
                if self.debug:
                    print(f"Using top 3 recommendations with confidence scores: {filtered_recommendations['confidence'].tolist()}")
            
            # Sort by confidence
            sorted_recommendations = filtered_recommendations.sort_values('confidence', ascending=False)
            
            if self.debug and not sorted_recommendations.empty:
                print(f"Top recommendation after sorting:")
                print(sorted_recommendations.iloc[0])
            
            return sorted_recommendations
        else:
            if self.debug:
                print(f"No recommendations generated")
            
            return pd.DataFrame()
    
    def get_underlying_price(self, symbol):
        """
        Get the current price of the underlying asset with caching
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            float: Current price
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_price"
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 60:  # 1 minute cache for prices
                    if self.debug:
                        print(f"Loading {symbol} price from cache (age: {file_age:.1f}s)")
                    try:
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                            return data.get('price')
                    except Exception as e:
                        print(f"Error loading price cache: {str(e)}")
            
            # Fetch fresh price
            quote = self.data_collector.get_quote(symbol)
            if quote and 'lastPrice' in quote:
                price = quote['lastPrice']
                
                # Cache the price
                try:
                    with open(cache_file, 'w') as f:
                        json.dump({'price': price, 'timestamp': time.time()}, f)
                except Exception as e:
                    print(f"Error caching price: {str(e)}")
                
                return price
            else:
                print("No underlying price available")
                return None
        except Exception as e:
            print(f"Error retrieving underlying price: {str(e)}")
            return None
