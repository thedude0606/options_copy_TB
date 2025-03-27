"""
Enhanced recommendation engine module for options recommendation platform.
Implements Phase 3 improvements: dynamic weighting system, confidence score calibration,
and strategy-specific scoring models.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import pickle
import json
import math
from app.indicators.technical_indicators import TechnicalIndicators
from app.analysis.options_analysis import OptionsAnalysis
from app.indicators.multi_timeframe_analyzer import MultiTimeframeAnalyzer

class RecommendationEngine:
    """
    Enhanced class to generate options trading recommendations with Phase 3 improvements:
    - Dynamic weighting system that adapts to market conditions
    - Confidence score calibration for improved reliability
    - Strategy-specific scoring models for different trading approaches
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
        self.multi_timeframe_analyzer = None  # Will be initialized when needed
        
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
        
        # Initialize market regime tracking
        self.current_volatility_regime = 'normal'
        self.trend_strength = 'moderate'
        self.market_event_impact = 'low'
        
        # Base weights for scoring factors (will be dynamically adjusted)
        self.base_weights = {
            'market_direction': 20,
            'probability_of_profit': 20,
            'risk_reward_ratio': 15,
            'delta': 15,
            'days_to_expiration': 10,
            'liquidity': 10,
            'bid_ask_spread': 10
        }
        
        # Current dynamic weights (initialized to base weights)
        self.current_weights = self.base_weights.copy()
        
        # Strategy-specific base weights
        self.strategy_weights = {
            'directional': {
                'market_direction': 30,
                'probability_of_profit': 15,
                'risk_reward_ratio': 15,
                'delta': 20,
                'days_to_expiration': 5,
                'liquidity': 10,
                'bid_ask_spread': 5
            },
            'income': {
                'market_direction': 10,
                'probability_of_profit': 30,
                'risk_reward_ratio': 15,
                'delta': 10,
                'days_to_expiration': 15,
                'liquidity': 10,
                'bid_ask_spread': 10
            },
            'volatility': {
                'market_direction': 5,
                'probability_of_profit': 15,
                'risk_reward_ratio': 20,
                'delta': 10,
                'days_to_expiration': 10,
                'liquidity': 15,
                'bid_ask_spread': 25
            }
        }
        
        # Confidence calibration parameters
        self.calibration_params = {
            'sigmoid_center': 50,  # Center point of sigmoid function
            'sigmoid_steepness': 0.1,  # Steepness of sigmoid curve
            'regime_scaling': {
                'high_volatility': 0.8,  # Reduce confidence in high volatility
                'normal_volatility': 1.0,  # Normal scaling
                'low_volatility': 1.2    # Increase confidence in low volatility
            },
            'trend_scaling': {
                'strong': 1.2,    # Increase confidence in strong trends
                'moderate': 1.0,  # Normal scaling
                'weak': 0.8       # Reduce confidence in weak trends
            },
            'event_impact_penalty': {
                'high': 0.3,      # Significant penalty for high impact events
                'medium': 0.15,   # Moderate penalty
                'low': 0.0        # No penalty
            },
            'confluence_bonus': {
                'strong': 0.2,    # Significant bonus for strong confluence
                'moderate': 0.1,  # Moderate bonus
                'weak': 0.0       # No bonus
            }
        }
    
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
            
            # Update market regime based on context
            self._update_market_regime(market_context)
            
            if self.debug:
                print(f"Current market regime: Volatility={self.current_volatility_regime}, Trend={self.trend_strength}, Event Impact={self.market_event_impact}")
            
            # Update dynamic weights based on market regime
            self._update_dynamic_weights(strategy_types)
            
            if self.debug:
                print(f"Dynamic weights updated: {self.current_weights}")
            
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
            
            # Get trading signals
            signals = self._generate_signals(indicators, market_context)
            
            if self.debug:
                print(f"Trading signals: Bullish={signals['bullish']:.2f}, Bearish={signals['bearish']:.2f}, Neutral={signals['neutral']:.2f}")
                print(f"Signal confidence: {signals['confidence']:.2f}")
                print(f"Signal details: {signals['signal_details'][:3]}...")
            
            # Score options with enhanced algorithm
            recommendations = self._score_options(options_analysis, signals, confidence_threshold, strategy_types)
            
            if self.debug:
                print(f"Generated {len(recommendations)} recommendations")
                if not recommendations.empty:
                    print(f"Top recommendation: {recommendations.iloc[0]['symbol']} ({recommendations.iloc[0]['strategy']})")
                    print(f"Confidence: {recommendations.iloc[0]['confidence']:.2f}")
            
            return recommendations
        
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            import traceback
            traceback.print_exc()
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
            if self.debug:
                print(f"Caching {cache_key} data")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"Error caching data: {str(e)}")
        
        return data
    
    def _fetch_historical_data(self, symbol, lookback_days=30):
        """
        Fetch historical price data for a symbol
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Historical price data
        """
        if self.debug:
            print(f"Fetching historical data for {symbol} (lookback: {lookback_days} days)")
        
        try:
            # Calculate period based on lookback days
            if lookback_days <= 5:
                period_type = 'day'
                period = 5
                frequency_type = 'minute'
                frequency = 5
            elif lookback_days <= 10:
                period_type = 'day'
                period = 10
                frequency_type = 'minute'
                frequency = 15
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
            
            # Fetch data from data collector
            historical_data = self.data_collector.get_historical_data(
                symbol=symbol,
                period_type=period_type,
                period=period,
                frequency_type=frequency_type,
                frequency=frequency
            )
            
            if self.debug:
                print(f"Fetched {len(historical_data)} historical data points")
            
            return historical_data
            
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_options_data(self, symbol):
        """
        Fetch options chain data for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            pd.DataFrame: Options chain data
        """
        if self.debug:
            print(f"Fetching options data for {symbol}")
        
        try:
            # Fetch options chain from data collector
            options_data = self.data_collector.get_options_chain(symbol)
            
            if self.debug:
                print(f"Fetched {len(options_data)} options contracts")
            
            return options_data
            
        except Exception as e:
            print(f"Error fetching options data: {str(e)}")
            return pd.DataFrame()
    
    def _get_market_context(self, symbol):
        """
        Get market context data for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Market context data
        """
        if self.debug:
            print(f"Getting market context for {symbol}")
        
        try:
            # Initialize context with default values
            context = {
                'vix': 15.0,  # Default VIX value
                'adx': 20.0,  # Default ADX value
                'market_trend': 'neutral',
                'sector_trend': 'neutral',
                'upcoming_events': []
            }
            
            # Try to get VIX data
            try:
                vix_data = self.data_collector.get_historical_data(
                    symbol='VIX',
                    period_type='day',
                    period=5,
                    frequency_type='daily',
                    frequency=1
                )
                
                if not vix_data.empty:
                    context['vix'] = vix_data['close'].iloc[-1]
            except Exception as e:
                if self.debug:
                    print(f"Error getting VIX data: {str(e)}")
            
            # Try to get ADX for the symbol
            try:
                historical_data = self.data_collector.get_historical_data(
                    symbol=symbol,
                    period_type='month',
                    period=1,
                    frequency_type='daily',
                    frequency=1
                )
                
                if not historical_data.empty:
                    adx = self.technical_indicators.calculate_adx(historical_data, period=14)
                    if not pd.isna(adx.iloc[-1]):
                        context['adx'] = adx.iloc[-1]
            except Exception as e:
                if self.debug:
                    print(f"Error calculating ADX: {str(e)}")
            
            # Determine market trend based on SPY
            try:
                spy_data = self.data_collector.get_historical_data(
                    symbol='SPY',
                    period_type='month',
                    period=1,
                    frequency_type='daily',
                    frequency=1
                )
                
                if not spy_data.empty:
                    # Calculate 20-day SMA
                    spy_data['sma20'] = spy_data['close'].rolling(window=20).mean()
                    
                    # Determine trend
                    last_close = spy_data['close'].iloc[-1]
                    last_sma = spy_data['sma20'].iloc[-1]
                    
                    if last_close > last_sma * 1.02:
                        context['market_trend'] = 'bullish'
                    elif last_close < last_sma * 0.98:
                        context['market_trend'] = 'bearish'
                    else:
                        context['market_trend'] = 'neutral'
            except Exception as e:
                if self.debug:
                    print(f"Error determining market trend: {str(e)}")
            
            # Add any upcoming earnings or events
            # This would typically come from a calendar API or data provider
            # For now, we'll just use a placeholder
            context['upcoming_events'] = [
                {'event': 'earnings', 'date': None, 'impact': 'low'}
            ]
            
            return context
            
        except Exception as e:
            print(f"Error getting market context: {str(e)}")
            return {}
    
    def _calculate_indicators(self, historical_data):
        """
        Calculate technical indicators for historical data
        
        Args:
            historical_data (pd.DataFrame): Historical price data
            
        Returns:
            dict: Dictionary of calculated indicators
        """
        if self.debug:
            print(f"Calculating technical indicators")
        
        try:
            indicators = {}
            
            # Basic price indicators
            indicators['close'] = historical_data['close']
            indicators['high'] = historical_data['high']
            indicators['low'] = historical_data['low']
            indicators['volume'] = historical_data['volume']
            
            # Moving averages
            indicators['sma20'] = self.technical_indicators.calculate_sma(historical_data, period=20)
            indicators['sma50'] = self.technical_indicators.calculate_sma(historical_data, period=50)
            indicators['sma200'] = self.technical_indicators.calculate_sma(historical_data, period=200)
            indicators['ema12'] = self.technical_indicators.calculate_ema(historical_data, period=12)
            indicators['ema26'] = self.technical_indicators.calculate_ema(historical_data, period=26)
            
            # Momentum indicators
            indicators['rsi'] = self.technical_indicators.calculate_rsi(historical_data)
            indicators['macd'] = self.technical_indicators.calculate_macd(historical_data)
            indicators['macd_signal'] = self.technical_indicators.calculate_macd_signal(historical_data)
            indicators['macd_histogram'] = self.technical_indicators.calculate_macd_histogram(historical_data)
            
            # Volatility indicators
            indicators['bollinger_upper'], indicators['bollinger_middle'], indicators['bollinger_lower'] = \
                self.technical_indicators.calculate_bollinger_bands(historical_data)
            indicators['atr'] = self.technical_indicators.calculate_atr(historical_data)
            
            # Trend indicators
            try:
                indicators['adx'] = self.technical_indicators.calculate_adx(historical_data)
            except Exception as e:
                if self.debug:
                    print(f"Error calculating ADX: {str(e)}")
                indicators['adx'] = pd.Series(20.0, index=historical_data.index)  # Default value
            
            # Volume indicators
            indicators['obv'] = self.technical_indicators.calculate_obv(historical_data)
            
            # Enhanced indicators from Phase 1
            try:
                indicators['cmo'] = self.technical_indicators.calculate_cmo(historical_data)
                indicators['stoch_rsi_k'], indicators['stoch_rsi_d'] = self.technical_indicators.calculate_stochastic_rsi(historical_data)
                indicators['adl'] = self.technical_indicators.calculate_adl(historical_data)
                indicators['ama'] = self.technical_indicators.calculate_adaptive_moving_average(historical_data)
                indicators['volatility_regime'] = self.technical_indicators.identify_volatility_regime(historical_data)
            except Exception as e:
                if self.debug:
                    print(f"Error calculating enhanced indicators: {str(e)}")
            
            # Candlestick patterns
            indicators['patterns'] = self.technical_indicators.identify_candlestick_patterns(historical_data)
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return {}
    
    def _analyze_options(self, options_data):
        """
        Analyze options data to calculate additional metrics
        
        Args:
            options_data (pd.DataFrame): Options chain data
            
        Returns:
            pd.DataFrame: Analyzed options data
        """
        if self.debug:
            print(f"Analyzing options data")
        
        try:
            # Make a copy to avoid modifying the original
            analyzed_data = options_data.copy()
            
            # Skip if empty
            if analyzed_data.empty:
                return analyzed_data
            
            # Calculate probability of profit if not already present
            if 'probabilityOfProfit' not in analyzed_data.columns:
                analyzed_data['probabilityOfProfit'] = self.options_analysis.calculate_probability_of_profit(analyzed_data)
            
            # Calculate risk-reward ratio if not already present
            if 'riskRewardRatio' not in analyzed_data.columns:
                analyzed_data['riskRewardRatio'] = self.options_analysis.calculate_risk_reward_ratio(analyzed_data)
            
            # Calculate potential return if not already present
            if 'potentialReturn' not in analyzed_data.columns:
                analyzed_data['potentialReturn'] = self.options_analysis.calculate_potential_return(analyzed_data)
            
            # Calculate liquidity score if not already present
            if 'liquidityScore' not in analyzed_data.columns:
                analyzed_data['liquidityScore'] = self.options_analysis.calculate_liquidity_score(analyzed_data)
            
            # Calculate IV rank if not already present
            if 'ivRank' not in analyzed_data.columns:
                analyzed_data['ivRank'] = self.options_analysis.calculate_iv_rank(analyzed_data)
            
            # Calculate expected move if not already present
            if 'expectedMove' not in analyzed_data.columns:
                analyzed_data['expectedMove'] = self.options_analysis.calculate_expected_move(analyzed_data)
            
            return analyzed_data
            
        except Exception as e:
            print(f"Error analyzing options: {str(e)}")
            return options_data
    
    def _generate_signals(self, indicators, market_context):
        """
        Generate trading signals from technical indicators and market context
        
        Args:
            indicators (dict): Dictionary of technical indicators
            market_context (dict): Market context data
            
        Returns:
            dict: Dictionary of trading signals
        """
        if self.debug:
            print(f"Generating trading signals")
        
        try:
            # Initialize signal counters
            bullish_signals = 0
            bearish_signals = 0
            neutral_signals = 0
            signal_details = []
            
            # Skip if indicators are empty
            if not indicators:
                return {
                    'bullish': 0,
                    'bearish': 0,
                    'neutral': 1,
                    'confidence': 0,
                    'signal_details': ['No indicators available']
                }
            
            # Get the last values for each indicator
            last_close = indicators['close'].iloc[-1] if 'close' in indicators and not indicators['close'].empty else 0
            last_sma20 = indicators['sma20'].iloc[-1] if 'sma20' in indicators and not indicators['sma20'].empty else 0
            last_sma50 = indicators['sma50'].iloc[-1] if 'sma50' in indicators and not indicators['sma50'].empty else 0
            last_sma200 = indicators['sma200'].iloc[-1] if 'sma200' in indicators and not indicators['sma200'].empty else 0
            last_ema12 = indicators['ema12'].iloc[-1] if 'ema12' in indicators and not indicators['ema12'].empty else 0
            last_ema26 = indicators['ema26'].iloc[-1] if 'ema26' in indicators and not indicators['ema26'].empty else 0
            last_rsi = indicators['rsi'].iloc[-1] if 'rsi' in indicators and not indicators['rsi'].empty else 50
            last_macd = indicators['macd'].iloc[-1] if 'macd' in indicators and not indicators['macd'].empty else 0
            last_macd_signal = indicators['macd_signal'].iloc[-1] if 'macd_signal' in indicators and not indicators['macd_signal'].empty else 0
            last_macd_histogram = indicators['macd_histogram'].iloc[-1] if 'macd_histogram' in indicators and not indicators['macd_histogram'].empty else 0
            last_bb_upper = indicators['bollinger_upper'].iloc[-1] if 'bollinger_upper' in indicators and not indicators['bollinger_upper'].empty else 0
            last_bb_middle = indicators['bollinger_middle'].iloc[-1] if 'bollinger_middle' in indicators and not indicators['bollinger_middle'].empty else 0
            last_bb_lower = indicators['bollinger_lower'].iloc[-1] if 'bollinger_lower' in indicators and not indicators['bollinger_lower'].empty else 0
            last_adx = indicators['adx'].iloc[-1] if 'adx' in indicators and not indicators['adx'].empty else 0
            
            # Enhanced indicators from Phase 1
            last_cmo = indicators['cmo'].iloc[-1] if 'cmo' in indicators and not indicators['cmo'].empty else 0
            last_stoch_rsi_k = indicators['stoch_rsi_k'].iloc[-1] if 'stoch_rsi_k' in indicators and not indicators['stoch_rsi_k'].empty else 50
            last_stoch_rsi_d = indicators['stoch_rsi_d'].iloc[-1] if 'stoch_rsi_d' in indicators and not indicators['stoch_rsi_d'].empty else 50
            last_volatility_regime = indicators['volatility_regime'].iloc[-1] if 'volatility_regime' in indicators and not indicators['volatility_regime'].empty else 'normal'
            
            # Check for candlestick patterns
            patterns = indicators['patterns'].iloc[-1] if 'patterns' in indicators and not indicators['patterns'].empty else []
            
            # Moving average signals
            if last_close > last_sma20:
                bullish_signals += 1
                signal_details.append(f"Price above 20-day SMA: {last_close:.2f} > {last_sma20:.2f}")
            elif last_close < last_sma20:
                bearish_signals += 1
                signal_details.append(f"Price below 20-day SMA: {last_close:.2f} < {last_sma20:.2f}")
            else:
                neutral_signals += 1
                signal_details.append(f"Price at 20-day SMA: {last_close:.2f} ≈ {last_sma20:.2f}")
            
            if last_close > last_sma50:
                bullish_signals += 1
                signal_details.append(f"Price above 50-day SMA: {last_close:.2f} > {last_sma50:.2f}")
            elif last_close < last_sma50:
                bearish_signals += 1
                signal_details.append(f"Price below 50-day SMA: {last_close:.2f} < {last_sma50:.2f}")
            else:
                neutral_signals += 1
                signal_details.append(f"Price at 50-day SMA: {last_close:.2f} ≈ {last_sma50:.2f}")
            
            if last_close > last_sma200:
                bullish_signals += 1
                signal_details.append(f"Price above 200-day SMA: {last_close:.2f} > {last_sma200:.2f}")
            elif last_close < last_sma200:
                bearish_signals += 1
                signal_details.append(f"Price below 200-day SMA: {last_close:.2f} < {last_sma200:.2f}")
            else:
                neutral_signals += 1
                signal_details.append(f"Price at 200-day SMA: {last_close:.2f} ≈ {last_sma200:.2f}")
            
            # EMA crossover
            if last_ema12 > last_ema26:
                bullish_signals += 1
                signal_details.append(f"EMA12 above EMA26: {last_ema12:.2f} > {last_ema26:.2f}")
            elif last_ema12 < last_ema26:
                bearish_signals += 1
                signal_details.append(f"EMA12 below EMA26: {last_ema12:.2f} < {last_ema26:.2f}")
            else:
                neutral_signals += 1
                signal_details.append(f"EMA12 at EMA26: {last_ema12:.2f} ≈ {last_ema26:.2f}")
            
            # RSI signals
            if last_rsi > 70:
                bearish_signals += 1
                signal_details.append(f"RSI overbought: {last_rsi:.1f}")
            elif last_rsi < 30:
                bullish_signals += 1
                signal_details.append(f"RSI oversold: {last_rsi:.1f}")
            else:
                neutral_signals += 1
                signal_details.append(f"RSI neutral: {last_rsi:.1f}")
            
            # MACD signals
            if last_macd > last_macd_signal:
                bullish_signals += 1
                signal_details.append(f"MACD bullish crossover")
            elif last_macd < last_macd_signal:
                bearish_signals += 1
                signal_details.append(f"MACD bearish crossover")
            else:
                neutral_signals += 1
                signal_details.append(f"MACD neutral")
            
            if last_macd_histogram > 0:
                bullish_signals += 1
                signal_details.append(f"MACD histogram positive: {last_macd_histogram:.4f}")
            elif last_macd_histogram < 0:
                bearish_signals += 1
                signal_details.append(f"MACD histogram negative: {last_macd_histogram:.4f}")
            else:
                neutral_signals += 1
                signal_details.append(f"MACD histogram neutral: {last_macd_histogram:.4f}")
            
            # Bollinger Bands signals
            if last_close > last_bb_upper:
                bearish_signals += 1
                signal_details.append(f"Price above upper Bollinger Band: {last_close:.2f} > {last_bb_upper:.2f}")
            elif last_close < last_bb_lower:
                bullish_signals += 1
                signal_details.append(f"Price below lower Bollinger Band: {last_close:.2f} < {last_bb_lower:.2f}")
            else:
                neutral_signals += 1
                signal_details.append(f"Price within Bollinger Bands")
            
            # ADX signals
            if last_adx > 25:
                # Strong trend, direction determined by other indicators
                if bullish_signals > bearish_signals:
                    bullish_signals += 1
                    signal_details.append(f"Strong bullish trend detected: ADX={last_adx:.1f}")
                elif bearish_signals > bullish_signals:
                    bearish_signals += 1
                    signal_details.append(f"Strong bearish trend detected: ADX={last_adx:.1f}")
                else:
                    neutral_signals += 1
                    signal_details.append(f"Strong but unclear trend: ADX={last_adx:.1f}")
            else:
                neutral_signals += 1
                signal_details.append(f"Weak trend: ADX={last_adx:.1f}")
            
            # Enhanced indicator signals from Phase 1
            # CMO signals
            if last_cmo > 50:
                bearish_signals += 1
                signal_details.append(f"CMO overbought: {last_cmo:.1f}")
            elif last_cmo < -50:
                bullish_signals += 1
                signal_details.append(f"CMO oversold: {last_cmo:.1f}")
            else:
                neutral_signals += 1
                signal_details.append(f"CMO neutral: {last_cmo:.1f}")
            
            # Stochastic RSI signals
            if last_stoch_rsi_k > 80:
                bearish_signals += 1
                signal_details.append(f"Stochastic RSI overbought: K={last_stoch_rsi_k}, D={last_stoch_rsi_d}")
            elif last_stoch_rsi_k < 20:
                bullish_signals += 1
                signal_details.append(f"Stochastic RSI oversold: K={last_stoch_rsi_k}, D={last_stoch_rsi_d}")
            elif last_stoch_rsi_k > last_stoch_rsi_d:
                bullish_signals += 1
                signal_details.append(f"Stochastic RSI bullish: K={last_stoch_rsi_k}, D={last_stoch_rsi_d}")
            elif last_stoch_rsi_k < last_stoch_rsi_d:
                bearish_signals += 1
                signal_details.append(f"Stochastic RSI bearish: K={last_stoch_rsi_k}, D={last_stoch_rsi_d}")
            else:
                neutral_signals += 1
                signal_details.append(f"Stochastic RSI neutral: K={last_stoch_rsi_k}, D={last_stoch_rsi_d}")
            
            # Volatility regime signals
            if last_volatility_regime == 'high':
                # In high volatility, be more cautious
                neutral_signals += 1
                signal_details.append(f"High volatility regime detected")
            elif last_volatility_regime == 'low':
                # In low volatility, trend following is more reliable
                if bullish_signals > bearish_signals:
                    bullish_signals += 1
                    signal_details.append(f"Low volatility bullish trend")
                elif bearish_signals > bullish_signals:
                    bearish_signals += 1
                    signal_details.append(f"Low volatility bearish trend")
                else:
                    neutral_signals += 1
                    signal_details.append(f"Low volatility neutral trend")
            else:
                signal_details.append(f"Normal volatility regime")
            
            # Candlestick pattern signals
            bullish_patterns = [p for p in patterns if p in ['Hammer', 'Bullish Engulfing', 'Morning Star', 'Piercing Line', 'Bullish Harami']]
            bearish_patterns = [p for p in patterns if p in ['Shooting Star', 'Bearish Engulfing', 'Evening Star', 'Dark Cloud Cover', 'Bearish Harami']]
            
            if bullish_patterns:
                bullish_signals += len(bullish_patterns)
                signal_details.append(f"Bullish patterns: {', '.join(bullish_patterns)}")
            
            if bearish_patterns:
                bearish_signals += len(bearish_patterns)
                signal_details.append(f"Bearish patterns: {', '.join(bearish_patterns)}")
            
            # Market context signals
            if 'market_trend' in market_context:
                if market_context['market_trend'] == 'bullish':
                    bullish_signals += 1
                    signal_details.append(f"Bullish market trend")
                elif market_context['market_trend'] == 'bearish':
                    bearish_signals += 1
                    signal_details.append(f"Bearish market trend")
                else:
                    neutral_signals += 1
                    signal_details.append(f"Neutral market trend")
            
            # Calculate confidence based on signal strength and consistency
            total_signals = bullish_signals + bearish_signals + neutral_signals
            if total_signals > 0:
                max_signal = max(bullish_signals, bearish_signals, neutral_signals)
                signal_ratio = max_signal / total_signals
                confidence = signal_ratio * 0.8 + 0.2  # Scale to 0.2-1.0 range
            else:
                confidence = 0.2  # Minimum confidence
            
            # Create signal summary
            signals = {
                'bullish': bullish_signals,
                'bearish': bearish_signals,
                'neutral': neutral_signals,
                'confidence': confidence,
                'signal_details': signal_details
            }
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return {
                'bullish': 0,
                'bearish': 0,
                'neutral': 1,
                'confidence': 0,
                'signal_details': [f"Error: {str(e)}"]
            }
    
    def _update_market_regime(self, market_context):
        """
        Update market regime based on market context
        
        Args:
            market_context (dict): Market context data
        """
        # Update volatility regime
        if 'vix' in market_context:
            vix = market_context['vix']
            if vix > 25:
                self.current_volatility_regime = 'high'
            elif vix < 15:
                self.current_volatility_regime = 'low'
            else:
                self.current_volatility_regime = 'normal'
        
        # Update trend strength
        if 'adx' in market_context:
            adx = market_context['adx']
            if adx > 30:
                self.trend_strength = 'strong'
            elif adx > 20:
                self.trend_strength = 'moderate'
            else:
                self.trend_strength = 'weak'
        
        # Update market event impact
        if 'upcoming_events' in market_context:
            events = market_context['upcoming_events']
            if any(e['impact'] == 'high' for e in events):
                self.market_event_impact = 'high'
            elif any(e['impact'] == 'medium' for e in events):
                self.market_event_impact = 'medium'
            else:
                self.market_event_impact = 'low'
    
    def _update_dynamic_weights(self, strategy_types=None):
        """
        Update dynamic weights based on market regime and selected strategies
        
        Args:
            strategy_types (list): Optional list of strategy types to consider
        """
        # Start with base weights
        self.current_weights = self.base_weights.copy()
        
        # If specific strategies are selected, blend their weights
        if strategy_types:
            blended_weights = {k: 0 for k in self.base_weights.keys()}
            strategy_count = len(strategy_types)
            
            for strategy in strategy_types:
                if strategy in self.strategy_weights:
                    for factor, weight in self.strategy_weights[strategy].items():
                        blended_weights[factor] += weight / strategy_count
            
            # Replace base weights with blended strategy weights
            self.current_weights = blended_weights
        
        # Adjust weights based on volatility regime
        if self.current_volatility_regime == 'high':
            # In high volatility, favor probability of profit and risk-reward
            self.current_weights['probability_of_profit'] *= 1.3
            self.current_weights['risk_reward_ratio'] *= 1.2
            self.current_weights['market_direction'] *= 0.8
            self.current_weights['days_to_expiration'] *= 0.7
        elif self.current_volatility_regime == 'low':
            # In low volatility, favor market direction and days to expiration
            self.current_weights['market_direction'] *= 1.2
            self.current_weights['days_to_expiration'] *= 1.3
            self.current_weights['probability_of_profit'] *= 0.9
            self.current_weights['bid_ask_spread'] *= 0.8
        
        # Adjust weights based on trend strength
        if self.trend_strength == 'strong':
            # In strong trends, favor market direction and delta
            self.current_weights['market_direction'] *= 1.3
            self.current_weights['delta'] *= 1.2
            self.current_weights['bid_ask_spread'] *= 0.8
            self.current_weights['liquidity'] *= 0.9
        elif self.trend_strength == 'weak':
            # In weak trends, favor probability of profit and liquidity
            self.current_weights['probability_of_profit'] *= 1.2
            self.current_weights['liquidity'] *= 1.3
            self.current_weights['market_direction'] *= 0.7
            self.current_weights['delta'] *= 0.9
        
        # Normalize weights to sum to 100
        weight_sum = sum(self.current_weights.values())
        if weight_sum > 0:
            for factor in self.current_weights:
                self.current_weights[factor] = (self.current_weights[factor] / weight_sum) * 100
    
    def _calibrate_confidence_score(self, raw_score, strategy, confluence_strength='weak'):
        """
        Calibrate confidence score using sigmoid transformation and market regime adjustments
        
        Args:
            raw_score (float): Raw confidence score (0-100)
            strategy (str): Strategy type
            confluence_strength (str): Strength of signal confluence ('weak', 'moderate', 'strong')
            
        Returns:
            float: Calibrated confidence score (0-1)
        """
        # Apply sigmoid transformation for better distribution
        center = self.calibration_params['sigmoid_center']
        steepness = self.calibration_params['sigmoid_steepness']
        sigmoid_score = 1 / (1 + math.exp(-steepness * (raw_score - center)))
        
        # Apply market regime scaling
        regime_scale = self.calibration_params['regime_scaling'].get(
            f"{self.current_volatility_regime}_volatility", 1.0)
        
        # Apply trend strength scaling
        trend_scale = self.calibration_params['trend_scaling'].get(
            self.trend_strength, 1.0)
        
        # Apply event impact penalty
        event_penalty = self.calibration_params['event_impact_penalty'].get(
            self.market_event_impact, 0.0)
        
        # Apply confluence bonus
        confluence_bonus = self.calibration_params['confluence_bonus'].get(
            confluence_strength, 0.0)
        
        # Calculate final calibrated score
        calibrated_score = sigmoid_score * regime_scale * trend_scale
        
        # Apply penalty and bonus
        calibrated_score = max(0, min(1, calibrated_score - event_penalty + confluence_bonus))
        
        return calibrated_score
    
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
            
            # Determine primary strategy for this option
            primary_strategy = self._determine_primary_strategy(row, market_direction)
            
            # Get raw score using dynamic weights
            raw_score = self._calculate_raw_score(row, market_direction, bullish_score, bearish_score, neutral_score)
            
            if self.debug and idx == 0:
                print(f"  Raw score: {raw_score:.2f}")
            
            # Determine confluence strength based on signals
            confluence_strength = self._determine_confluence_strength(signals)
            
            # Calibrate confidence score
            confidence = self._calibrate_confidence_score(raw_score, primary_strategy, confluence_strength)
            
            if self.debug and idx == 0:
                print(f"  Calibrated confidence: {confidence:.2f}")
            
            # Get underlying price
            underlying_price = 0
            if 'underlyingPrice' in row and not pd.isna(row['underlyingPrice']):
                underlying_price = row['underlyingPrice']
            
            # Calculate entry price (mid price)
            entry_price = 0
            if 'bid' in row and 'ask' in row and not pd.isna(row['bid']) and not pd.isna(row['ask']):
                entry_price = (row['bid'] + row['ask']) / 2
            
            # Get days to expiration
            days = 0
            if 'daysToExpiration' in row and not pd.isna(row['daysToExpiration']):
                # Convert Timestamp to days if it's a Timestamp object
                if isinstance(row['daysToExpiration'], pd.Timedelta):
                    days = row['daysToExpiration'].days
                else:
                    # If it's already a number, use it directly
                    try:
                        days = float(row['daysToExpiration'])
                    except (ValueError, TypeError):
                        days = 0
            
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
                'daysToExpiration': days,
                'liquidityScore': row['liquidityScore'] if 'liquidityScore' in row else 0,
                'ivRank': row['ivRank'] if 'ivRank' in row else 0,
                'expectedMove': row['expectedMove'] if 'expectedMove' in row else 0,
                'rawScore': raw_score,
                'confidence': confidence,
                'marketDirection': market_direction,
                'strategy': primary_strategy,
                'signalDetails': signals['signal_details'],
                'marketRegime': {
                    'volatility': self.current_volatility_regime,
                    'trend': self.trend_strength,
                    'eventImpact': self.market_event_impact
                },
                'confluenceStrength': confluence_strength
            })
        
        # Convert to DataFrame and filter by confidence threshold
        recommendations_df = pd.DataFrame(scores)
        
        if self.debug:
            print(f"Created recommendations DataFrame with {len(scores)} rows")
            if not recommendations_df.empty:
                print(f"Recommendations columns: {recommendations_df.columns.tolist()}")
        
        # Filter by confidence threshold
        if not recommendations_df.empty:
            recommendations_df = recommendations_df[recommendations_df['confidence'] >= confidence_threshold]
            
            # Sort by confidence score (descending)
            recommendations_df = recommendations_df.sort_values(by='confidence', ascending=False)
            
            if self.debug:
                print(f"Filtered to {len(recommendations_df)} recommendations with confidence >= {confidence_threshold}")
        
        return recommendations_df
    
    def _determine_primary_strategy(self, option_data, market_direction):
        """
        Determine the primary strategy for an option based on its characteristics
        
        Args:
            option_data (pd.Series): Option data
            market_direction (str): Market direction ('bullish', 'bearish', 'neutral')
            
        Returns:
            str: Primary strategy
        """
        # Start with directional strategy based on market direction
        if market_direction == 'bullish' and option_data['optionType'] == 'CALL':
            strategy = "Directional Bullish"
        elif market_direction == 'bearish' and option_data['optionType'] == 'PUT':
            strategy = "Directional Bearish"
        elif market_direction == 'neutral':
            strategy = "Neutral"
        else:
            strategy = ""
        
        # Check for income strategy characteristics
        if 'theta' in option_data and not pd.isna(option_data['theta']) and option_data['theta'] < -0.01:
            if 'delta' in option_data and not pd.isna(option_data['delta']):
                delta = abs(option_data['delta'])
                if delta > 0.5:
                    strategy = "Income (High Delta)"
                else:
                    strategy = "Income (Low Delta)"
        
        # Check for volatility strategy characteristics
        elif 'vega' in option_data and not pd.isna(option_data['vega']) and option_data['vega'] > 0.05:
            if 'ivRank' in option_data and not pd.isna(option_data['ivRank']):
                if option_data['ivRank'] < 30:
                    strategy = "Long Volatility"
                elif option_data['ivRank'] > 70:
                    strategy = "Short Volatility"
                else:
                    strategy = "Volatility Neutral"
            else:
                strategy = "Volatility"
        
        # If no specific strategy identified, use balanced
        if not strategy:
            strategy = "Balanced"
        
        return strategy
    
    def _calculate_raw_score(self, option_data, market_direction, bullish_score, bearish_score, neutral_score):
        """
        Calculate raw score for an option using dynamic weights
        
        Args:
            option_data (pd.Series): Option data
            market_direction (str): Market direction ('bullish', 'bearish', 'neutral')
            bullish_score (float): Bullish signal score
            bearish_score (float): Bearish signal score
            neutral_score (float): Neutral signal score
            
        Returns:
            float: Raw score (0-100)
        """
        score = 0
        
        # Get dynamic weights
        weights = self.current_weights
        
        # Market direction score
        if market_direction == 'bullish' and option_data['optionType'] == 'CALL':
            score += bullish_score * weights['market_direction']
        elif market_direction == 'bearish' and option_data['optionType'] == 'PUT':
            score += bearish_score * weights['market_direction']
        elif market_direction == 'neutral':
            score += neutral_score * weights['market_direction']
        
        # Probability of profit score
        if 'probabilityOfProfit' in option_data and not pd.isna(option_data['probabilityOfProfit']):
            pop_score = option_data['probabilityOfProfit']
            score += pop_score * weights['probability_of_profit']
        
        # Risk-reward ratio score
        if 'riskRewardRatio' in option_data and not pd.isna(option_data['riskRewardRatio']):
            rr_ratio = option_data['riskRewardRatio']
            if rr_ratio > 0:
                rr_score = min(rr_ratio / 3, 1)  # Cap at 1
                score += rr_score * weights['risk_reward_ratio']
        
        # Delta score
        if 'delta' in option_data and not pd.isna(option_data['delta']):
            delta = abs(option_data['delta'])
            if 0.3 <= delta <= 0.7:
                delta_score = 1 - abs(delta - 0.5) / 0.5
            else:
                delta_score = 0.2  # Lower score for very low or high delta
            
            score += delta_score * weights['delta']
        
        # Days to expiration score
        if 'daysToExpiration' in option_data and not pd.isna(option_data['daysToExpiration']):
            # Convert Timestamp to days if it's a Timestamp object
            if isinstance(option_data['daysToExpiration'], pd.Timedelta):
                days = option_data['daysToExpiration'].days
            else:
                # If it's already a number, use it directly
                try:
                    days = float(option_data['daysToExpiration'])
                except (ValueError, TypeError):
                    days = 0
            
            if 20 <= days <= 60:
                dte_score = 1 - abs(days - 40) / 40
            else:
                dte_score = 0.2  # Lower score for very short or long DTE
            
            score += dte_score * weights['days_to_expiration']
        
        # Liquidity score
        if 'liquidityScore' in option_data and not pd.isna(option_data['liquidityScore']):
            liquidity_score = option_data['liquidityScore']
            score += liquidity_score * weights['liquidity']
        else:
            # Fallback liquidity scoring based on volume and open interest
            liquidity_score = 0
            if 'volume' in option_data and not pd.isna(option_data['volume']) and option_data['volume'] > 100:
                liquidity_score += 0.5
            if 'openInterest' in option_data and not pd.isna(option_data['openInterest']) and option_data['openInterest'] > 500:
                liquidity_score += 0.5
            
            score += liquidity_score * weights['liquidity']
        
        # Bid-ask spread score
        if 'bid' in option_data and 'ask' in option_data and not pd.isna(option_data['bid']) and not pd.isna(option_data['ask']):
            mid_price = (option_data['bid'] + option_data['ask']) / 2
            if mid_price > 0:
                spread_pct = (option_data['ask'] - option_data['bid']) / mid_price
                # Lower spread is better
                spread_score = max(0, 1 - (spread_pct * 10))  # Penalize spreads > 10%
                score += spread_score * weights['bid_ask_spread']
        
        return score
    
    def _determine_confluence_strength(self, signals):
        """
        Determine the strength of signal confluence
        
        Args:
            signals (dict): Dictionary of trading signals
            
        Returns:
            str: Confluence strength ('weak', 'moderate', 'strong')
        """
        # Check if confluence data is available
        if 'confluence_analysis' in signals:
            confluence = signals['confluence_analysis']
            
            # Get the maximum confluence percentage
            max_confluence = max(
                confluence.get('bullish_confluence', 0),
                confluence.get('bearish_confluence', 0)
            )
            
            # Determine strength based on confluence percentage
            if max_confluence >= 0.7:
                return 'strong'
            elif max_confluence >= 0.5:
                return 'moderate'
        
        # If no confluence data or low confluence, analyze signal consistency
        bullish = signals['bullish']
        bearish = signals['bearish']
        neutral = signals['neutral']
        total = bullish + bearish + neutral
        
        if total > 0:
            max_signal = max(bullish, bearish)
            signal_ratio = max_signal / total
            
            if signal_ratio >= 0.7:
                return 'strong'
            elif signal_ratio >= 0.5:
                return 'moderate'
        
        return 'weak'
