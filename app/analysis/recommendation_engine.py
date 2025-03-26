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
import logging
from app.indicators.technical_indicators import TechnicalIndicators
from app.analysis.options_analysis import OptionsAnalysis

# Configure logging for recommendation engine
logger = logging.getLogger('recommendation_engine')
logger.setLevel(logging.INFO)

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
            if self.debug:
                logger.info(f"\n=== RECOMMENDATION ENGINE DEBUG ===")
                logger.info(f"Generating recommendations for {symbol}")
                logger.info(f"Lookback days: {lookback_days}")
                logger.info(f"Confidence threshold: {confidence_threshold}")
                logger.info(f"Strategy types: {strategy_types if strategy_types else 'All'}")
            
            # Check cache for historical data
            historical_data = self._get_cached_data(f"{symbol}_historical", self._fetch_historical_data, 
                                                   symbol=symbol, lookback_days=lookback_days)
            
            if self.debug:
                logger.info(f"Historical data shape: {historical_data.shape if not historical_data.empty else 'Empty'}")
            
            if historical_data.empty:
                logger.warning(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Check cache for options data
            options_data = self._get_cached_data(f"{symbol}_options", self._fetch_options_data, symbol=symbol)
            
            if self.debug:
                logger.info(f"Options data shape: {options_data.shape if not options_data.empty else 'Empty'}")
                if not options_data.empty:
                    logger.info(f"Options data columns: {options_data.columns.tolist()}")
                    logger.info(f"Sample option data (first row):")
                    logger.info(options_data.iloc[0])
            
            if options_data.empty:
                logger.warning(f"No options data available for {symbol}")
                return pd.DataFrame()
            
            # Get market context data
            market_context = self._get_market_context(symbol)
            
            if self.debug:
                logger.info(f"Market context: {market_context}")
            
            # Calculate technical indicators with enhanced set
            if self.debug:
                logger.info(f"Calculating technical indicators...")
            
            indicators = self._calculate_indicators(historical_data)
            
            if self.debug:
                logger.info(f"Indicators calculated: {list(indicators.keys())}")
            
            # Calculate options Greeks and probabilities
            if self.debug:
                logger.info(f"Analyzing options data...")
            
            options_analysis = self._analyze_options(options_data)
            
            if self.debug:
                logger.info(f"Options analysis shape: {options_analysis.shape if not options_analysis.empty else 'Empty'}")
                if not options_analysis.empty:
                    logger.info(f"Options analysis columns: {options_analysis.columns.tolist()}")
            
            # Generate signals based on technical indicators and market context
            if self.debug:
                logger.info(f"Generating signals from indicators and market context...")
            
            signals = self._generate_signals(indicators, market_context)
            
            if self.debug:
                logger.info(f"Signal summary: Bullish={signals['bullish']}, Bearish={signals['bearish']}, Neutral={signals['neutral']}")
                logger.info(f"Signal details: {signals['signal_details']}")
            
            # Filter options based on signals and strategy types
            if self.debug:
                logger.info(f"Filtering options based on signals and strategy types...")
            
            filtered_options = self._filter_options(options_analysis, signals, strategy_types)
            
            if self.debug:
                logger.info(f"Filtered options shape: {filtered_options.shape if not filtered_options.empty else 'Empty'}")
            
            if filtered_options.empty:
                logger.warning(f"No suitable options found for {symbol} after filtering")
                return pd.DataFrame()
            
            # Score and rank options
            if self.debug:
                logger.info(f"Scoring and ranking options...")
            
            scored_options = self._score_options(filtered_options, signals, market_context)
            
            if self.debug:
                logger.info(f"Scored options shape: {scored_options.shape if not scored_options.empty else 'Empty'}")
                if not scored_options.empty:
                    logger.info(f"Top recommendation score: {scored_options['score'].max()}")
            
            # Filter by confidence threshold
            recommendations = scored_options[scored_options['confidence'] >= confidence_threshold]
            
            if self.debug:
                logger.info(f"Final recommendations shape: {recommendations.shape if not recommendations.empty else 'Empty'}")
                logger.info(f"Recommendations confidence range: {recommendations['confidence'].min() if not recommendations.empty else 0} - {recommendations['confidence'].max() if not recommendations.empty else 0}")
            
            if recommendations.empty:
                logger.warning(f"No recommendations meet the confidence threshold of {confidence_threshold}")
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return pd.DataFrame()
    
    def _get_cached_data(self, cache_key, fetch_func, **kwargs):
        """
        Get data from cache or fetch it if not available/expired
        
        Args:
            cache_key (str): Cache key
            fetch_func (function): Function to fetch data if not in cache
            **kwargs: Arguments to pass to fetch_func
            
        Returns:
            pd.DataFrame: Cached or freshly fetched data
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # Check if cache file exists and is not expired
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < self.cache_expiry:
                    if self.debug:
                        logger.info(f"Loading {cache_key} from cache (age: {file_age:.1f}s)")
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                else:
                    if self.debug:
                        logger.info(f"Cache expired for {cache_key} (age: {file_age:.1f}s)")
            else:
                if self.debug:
                    logger.info(f"No cache found for {cache_key}")
            
            # Fetch fresh data
            if self.debug:
                logger.info(f"Fetching fresh data for {cache_key}")
            data = fetch_func(**kwargs)
            
            # Save to cache
            if not data.empty:
                if self.debug:
                    logger.info(f"Saving {cache_key} to cache")
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            
            return data
        
        except Exception as e:
            logger.error(f"Error accessing cache for {cache_key}: {str(e)}")
            # Try to fetch fresh data as fallback
            try:
                return fetch_func(**kwargs)
            except Exception as e2:
                logger.error(f"Error fetching fresh data for {cache_key}: {str(e2)}")
                return pd.DataFrame()
    
    def _fetch_historical_data(self, symbol, lookback_days=30):
        """
        Fetch historical price data for a symbol
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            if self.debug:
                logger.info(f"Fetching {lookback_days} days of historical data for {symbol}")
            
            # Calculate period parameters based on lookback days
            if lookback_days <= 5:
                period_type, period = 'day', 5
                freq_type, freq = 'minute', 30
            elif lookback_days <= 30:
                period_type, period = 'month', 1
                freq_type, freq = 'daily', 1
            elif lookback_days <= 90:
                period_type, period = 'month', 3
                freq_type, freq = 'daily', 1
            elif lookback_days <= 180:
                period_type, period = 'month', 6
                freq_type, freq = 'daily', 1
            else:
                period_type, period = 'year', 1
                freq_type, freq = 'daily', 1
            
            if self.debug:
                logger.info(f"Using period_type={period_type}, period={period}, freq_type={freq_type}, freq={freq}")
            
            # Fetch historical data
            historical_data = self.data_collector.get_historical_data(
                symbol, 
                period_type=period_type, 
                period=period, 
                frequency_type=freq_type, 
                frequency=freq
            )
            
            if historical_data.empty:
                logger.warning(f"No historical data returned for {symbol}")
                return pd.DataFrame()
            
            if self.debug:
                logger.info(f"Fetched {len(historical_data)} historical data points for {symbol}")
                logger.info(f"Date range: {historical_data.index.min()} to {historical_data.index.max()}")
            
            return historical_data
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_options_data(self, symbol):
        """
        Fetch options data for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            pd.DataFrame: Options data
        """
        try:
            if self.debug:
                logger.info(f"Fetching options data for {symbol}")
            
            # Get option chain
            option_chain = self.data_collector.get_option_chain_with_underlying_price(symbol)
            
            if not option_chain or 'options' not in option_chain or not option_chain['options']:
                logger.warning(f"No options data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            options_data = pd.DataFrame(option_chain['options'])
            
            # Add underlying price
            underlying_price = option_chain.get('underlying_price', 0)
            if underlying_price <= 0:
                logger.warning(f"Invalid underlying price for {symbol}: {underlying_price}")
                # Try to get current price from another source
                quote_data = self.data_collector.get_quote(symbol)
                if quote_data and 'lastPrice' in quote_data:
                    underlying_price = quote_data['lastPrice']
                    logger.info(f"Using lastPrice from quote for {symbol}: {underlying_price}")
            
            options_data['underlying_price'] = underlying_price
            
            if self.debug:
                logger.info(f"Fetched {len(options_data)} options for {symbol}")
                logger.info(f"Underlying price: {underlying_price}")
                logger.info(f"Expiration dates: {options_data['expiration'].unique().tolist()}")
            
            return options_data
        
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_market_context(self, symbol):
        """
        Get market context data for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Market context data
        """
        try:
            if self.debug:
                logger.info(f"Getting market context for {symbol}")
            
            # Get market data
            market_data = self.data_collector.get_market_data()
            
            # Get symbol quote
            quote_data = self.data_collector.get_quote(symbol)
            
            # Calculate market context
            context = {
                'market_trend': 'neutral',  # Default
                'sector_trend': 'neutral',  # Default
                'volatility': 'normal',     # Default
                'liquidity': 'normal',      # Default
                'earnings_upcoming': False,  # Default
                'dividend_upcoming': False   # Default
            }
            
            # Determine market trend based on SPY
            if 'SPY' in market_data.get('quotes', {}):
                spy_change = market_data['quotes']['SPY'].get('netPercentChangeInDouble', 0)
                if spy_change > 1.0:
                    context['market_trend'] = 'bullish'
                elif spy_change < -1.0:
                    context['market_trend'] = 'bearish'
            
            # Determine volatility based on VIX
            if 'VIX' in market_data.get('quotes', {}):
                vix_value = market_data['quotes']['VIX'].get('lastPrice', 20)
                if vix_value > 30:
                    context['volatility'] = 'high'
                elif vix_value < 15:
                    context['volatility'] = 'low'
            
            # Determine liquidity based on volume
            if quote_data and 'totalVolume' in quote_data:
                avg_volume = quote_data.get('avgVolume', 0)
                current_volume = quote_data.get('totalVolume', 0)
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio > 1.5:
                        context['liquidity'] = 'high'
                    elif volume_ratio < 0.5:
                        context['liquidity'] = 'low'
            
            if self.debug:
                logger.info(f"Market context: {context}")
            
            return context
        
        except Exception as e:
            logger.error(f"Error getting market context for {symbol}: {str(e)}")
            return {
                'market_trend': 'neutral',
                'sector_trend': 'neutral',
                'volatility': 'normal',
                'liquidity': 'normal',
                'earnings_upcoming': False,
                'dividend_upcoming': False
            }
    
    def _calculate_indicators(self, historical_data):
        """
        Calculate technical indicators for historical data
        
        Args:
            historical_data (pd.DataFrame): Historical price data
            
        Returns:
            dict: Technical indicators
        """
        try:
            if historical_data.empty:
                logger.warning("Cannot calculate indicators: historical data is empty")
                return {}
            
            # Calculate basic indicators
            indicators = {}
            
            # Trend indicators
            indicators['sma_20'] = self.technical_indicators.sma(historical_data['close'], window=20)
            indicators['sma_50'] = self.technical_indicators.sma(historical_data['close'], window=50)
            indicators['sma_200'] = self.technical_indicators.sma(historical_data['close'], window=200)
            indicators['ema_12'] = self.technical_indicators.ema(historical_data['close'], window=12)
            indicators['ema_26'] = self.technical_indicators.ema(historical_data['close'], window=26)
            
            # Momentum indicators
            indicators['rsi'] = self.technical_indicators.rsi(historical_data['close'])
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.technical_indicators.macd(
                historical_data['close']
            )
            
            # Volatility indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.technical_indicators.bollinger_bands(
                historical_data['close']
            )
            indicators['atr'] = self.technical_indicators.average_true_range(
                historical_data['high'], historical_data['low'], historical_data['close']
            )
            
            # Volume indicators
            indicators['obv'] = self.technical_indicators.on_balance_volume(
                historical_data['close'], historical_data['volume']
            )
            
            # Get the most recent values for each indicator
            latest_indicators = {k: v.iloc[-1] if not pd.isna(v.iloc[-1]) else 0 for k, v in indicators.items()}
            
            # Add derived indicators
            latest_indicators['price'] = historical_data['close'].iloc[-1]
            latest_indicators['price_prev'] = historical_data['close'].iloc[-2] if len(historical_data) > 1 else latest_indicators['price']
            latest_indicators['volume'] = historical_data['volume'].iloc[-1]
            latest_indicators['volume_avg_20'] = historical_data['volume'].rolling(window=20).mean().iloc[-1]
            
            # Trend strength
            latest_indicators['above_sma_20'] = latest_indicators['price'] > latest_indicators['sma_20']
            latest_indicators['above_sma_50'] = latest_indicators['price'] > latest_indicators['sma_50']
            latest_indicators['above_sma_200'] = latest_indicators['price'] > latest_indicators['sma_200']
            latest_indicators['sma_20_slope'] = (latest_indicators['sma_20'] - indicators['sma_20'].iloc[-5]) / indicators['sma_20'].iloc[-5] if len(indicators['sma_20']) > 5 else 0
            
            # Momentum strength
            latest_indicators['rsi_overbought'] = latest_indicators['rsi'] > 70
            latest_indicators['rsi_oversold'] = latest_indicators['rsi'] < 30
            latest_indicators['macd_positive'] = latest_indicators['macd'] > 0
            latest_indicators['macd_crossover'] = (latest_indicators['macd'] > latest_indicators['macd_signal']) and (indicators['macd'].iloc[-2] <= indicators['macd_signal'].iloc[-2]) if len(indicators['macd']) > 2 else False
            latest_indicators['macd_crossunder'] = (latest_indicators['macd'] < latest_indicators['macd_signal']) and (indicators['macd'].iloc[-2] >= indicators['macd_signal'].iloc[-2]) if len(indicators['macd']) > 2 else False
            
            # Volatility indicators
            latest_indicators['bb_width'] = (latest_indicators['bb_upper'] - latest_indicators['bb_lower']) / latest_indicators['bb_middle']
            latest_indicators['bb_position'] = (latest_indicators['price'] - latest_indicators['bb_lower']) / (latest_indicators['bb_upper'] - latest_indicators['bb_lower']) if (latest_indicators['bb_upper'] - latest_indicators['bb_lower']) > 0 else 0.5
            
            # Volume indicators
            latest_indicators['volume_surge'] = latest_indicators['volume'] > (latest_indicators['volume_avg_20'] * 1.5)
            
            return latest_indicators
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def _analyze_options(self, options_data):
        """
        Analyze options data to calculate additional metrics
        
        Args:
            options_data (pd.DataFrame): Options data
            
        Returns:
            pd.DataFrame: Analyzed options data
        """
        try:
            if options_data.empty:
                logger.warning("Cannot analyze options: options data is empty")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            analyzed_options = options_data.copy()
            
            # Calculate mid price
            analyzed_options['mid_price'] = (analyzed_options['bid'] + analyzed_options['ask']) / 2
            
            # Calculate days to expiration
            analyzed_options['days_to_expiration'] = analyzed_options['expiration'].apply(
                lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days + 1
            )
            
            # Calculate moneyness
            analyzed_options['moneyness'] = analyzed_options.apply(
                lambda row: row['strike'] / row['underlying_price'] if row['underlying_price'] > 0 else 1,
                axis=1
            )
            
            # Categorize options by moneyness
            analyzed_options['moneyness_category'] = analyzed_options.apply(
                lambda row: self._categorize_moneyness(row['option_type'], row['moneyness']),
                axis=1
            )
            
            # Calculate probability of profit (approximate)
            analyzed_options['probability_of_profit'] = analyzed_options.apply(
                lambda row: self._calculate_probability_of_profit(row),
                axis=1
            )
            
            # Calculate risk-reward ratio
            analyzed_options['max_profit'] = analyzed_options.apply(
                lambda row: self._calculate_max_profit(row),
                axis=1
            )
            
            analyzed_options['max_loss'] = analyzed_options.apply(
                lambda row: self._calculate_max_loss(row),
                axis=1
            )
            
            analyzed_options['risk_reward_ratio'] = analyzed_options.apply(
                lambda row: row['max_profit'] / row['max_loss'] if row['max_loss'] > 0 else 0,
                axis=1
            )
            
            # Calculate potential return percentage
            analyzed_options['potential_return_pct'] = analyzed_options.apply(
                lambda row: (row['max_profit'] / row['max_loss']) * 100 if row['max_loss'] > 0 else 0,
                axis=1
            )
            
            # Calculate liquidity score (0-1)
            analyzed_options['liquidity_score'] = analyzed_options.apply(
                lambda row: min(1.0, (row['volume'] / 1000) * (1 / (1 + row['ask'] - row['bid']))),
                axis=1
            )
            
            return analyzed_options
        
        except Exception as e:
            logger.error(f"Error analyzing options: {str(e)}")
            return pd.DataFrame()
    
    def _categorize_moneyness(self, option_type, moneyness):
        """
        Categorize option by moneyness
        
        Args:
            option_type (str): Option type (CALL or PUT)
            moneyness (float): Strike price / underlying price
            
        Returns:
            str: Moneyness category (ITM, ATM, OTM)
        """
        if option_type == 'CALL':
            if moneyness < 0.95:
                return 'ITM'
            elif moneyness > 1.05:
                return 'OTM'
            else:
                return 'ATM'
        else:  # PUT
            if moneyness > 1.05:
                return 'ITM'
            elif moneyness < 0.95:
                return 'OTM'
            else:
                return 'ATM'
    
    def _calculate_probability_of_profit(self, option):
        """
        Calculate approximate probability of profit for an option
        
        Args:
            option (pd.Series): Option data
            
        Returns:
            float: Probability of profit (0-1)
        """
        try:
            # Use delta as an approximation for probability
            if option['option_type'] == 'CALL':
                return max(0, min(1, option['delta']))
            else:  # PUT
                return max(0, min(1, abs(option['delta'])))
        except:
            return 0.5  # Default to 50% if calculation fails
    
    def _calculate_max_profit(self, option):
        """
        Calculate maximum potential profit for an option
        
        Args:
            option (pd.Series): Option data
            
        Returns:
            float: Maximum profit
        """
        try:
            if option['option_type'] == 'CALL':
                if option['moneyness_category'] == 'OTM':
                    # For OTM calls, max profit is theoretically unlimited
                    # Use a reasonable target of 100% move in underlying
                    target_price = option['underlying_price'] * 2
                    max_profit = target_price - option['strike'] - option['mid_price']
                    return max(0, max_profit)
                elif option['moneyness_category'] == 'ITM':
                    # For ITM calls, max profit is also theoretically unlimited
                    # Use a reasonable target of 50% move in underlying
                    target_price = option['underlying_price'] * 1.5
                    max_profit = target_price - option['strike'] - option['mid_price']
                    return max(0, max_profit)
                else:  # ATM
                    # For ATM calls, use a 25% move in underlying
                    target_price = option['underlying_price'] * 1.25
                    max_profit = target_price - option['strike'] - option['mid_price']
                    return max(0, max_profit)
            else:  # PUT
                if option['moneyness_category'] == 'OTM':
                    # For OTM puts, max profit is if underlying goes to 0
                    max_profit = option['strike'] - option['mid_price']
                    return max(0, max_profit)
                elif option['moneyness_category'] == 'ITM':
                    # For ITM puts, max profit is if underlying goes to 0
                    max_profit = option['strike'] - option['mid_price']
                    return max(0, max_profit)
                else:  # ATM
                    # For ATM puts, use a 25% drop in underlying
                    target_price = option['underlying_price'] * 0.75
                    max_profit = option['strike'] - target_price - option['mid_price']
                    return max(0, max_profit)
        except:
            return option['mid_price']  # Default to current price if calculation fails
    
    def _calculate_max_loss(self, option):
        """
        Calculate maximum potential loss for an option
        
        Args:
            option (pd.Series): Option data
            
        Returns:
            float: Maximum loss
        """
        try:
            # For long options, max loss is the premium paid
            return option['mid_price']
        except:
            return 1.0  # Default to $1 if calculation fails
    
    def _generate_signals(self, indicators, market_context):
        """
        Generate trading signals based on technical indicators and market context
        
        Args:
            indicators (dict): Technical indicators
            market_context (dict): Market context data
            
        Returns:
            dict: Trading signals
        """
        try:
            if not indicators:
                logger.warning("Cannot generate signals: indicators dictionary is empty")
                return {'bullish': 0, 'bearish': 0, 'neutral': 1, 'signal_details': {}}
            
            # Initialize signal scores
            bullish_score = 0
            bearish_score = 0
            signal_details = {}
            
            # Trend signals
            if indicators.get('above_sma_20', False):
                bullish_score += 1
                signal_details['above_sma_20'] = "Price is above 20-day SMA"
            else:
                bearish_score += 1
                signal_details['below_sma_20'] = "Price is below 20-day SMA"
            
            if indicators.get('above_sma_50', False):
                bullish_score += 1
                signal_details['above_sma_50'] = "Price is above 50-day SMA"
            else:
                bearish_score += 1
                signal_details['below_sma_50'] = "Price is below 50-day SMA"
            
            if indicators.get('above_sma_200', False):
                bullish_score += 2
                signal_details['above_sma_200'] = "Price is above 200-day SMA (bullish trend)"
            else:
                bearish_score += 2
                signal_details['below_sma_200'] = "Price is below 200-day SMA (bearish trend)"
            
            if indicators.get('sma_20_slope', 0) > 0.01:
                bullish_score += 1
                signal_details['sma_20_rising'] = "20-day SMA is rising"
            elif indicators.get('sma_20_slope', 0) < -0.01:
                bearish_score += 1
                signal_details['sma_20_falling'] = "20-day SMA is falling"
            
            # Momentum signals
            if indicators.get('rsi_overbought', False):
                bearish_score += 2
                signal_details['rsi_overbought'] = "RSI is overbought (>70)"
            elif indicators.get('rsi_oversold', False):
                bullish_score += 2
                signal_details['rsi_oversold'] = "RSI is oversold (<30)"
            
            if indicators.get('macd_positive', False):
                bullish_score += 1
                signal_details['macd_positive'] = "MACD is positive"
            else:
                bearish_score += 1
                signal_details['macd_negative'] = "MACD is negative"
            
            if indicators.get('macd_crossover', False):
                bullish_score += 2
                signal_details['macd_crossover'] = "MACD crossed above signal line (bullish)"
            elif indicators.get('macd_crossunder', False):
                bearish_score += 2
                signal_details['macd_crossunder'] = "MACD crossed below signal line (bearish)"
            
            # Volatility signals
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position > 0.8:
                bearish_score += 1
                signal_details['near_upper_bb'] = "Price is near upper Bollinger Band (potential resistance)"
            elif bb_position < 0.2:
                bullish_score += 1
                signal_details['near_lower_bb'] = "Price is near lower Bollinger Band (potential support)"
            
            bb_width = indicators.get('bb_width', 0)
            if bb_width > 0.1:
                signal_details['high_volatility'] = "Bollinger Bands are wide (high volatility)"
            elif bb_width < 0.05:
                signal_details['low_volatility'] = "Bollinger Bands are narrow (low volatility, potential breakout)"
            
            # Volume signals
            if indicators.get('volume_surge', False):
                if indicators.get('price', 0) > indicators.get('price_prev', 0):
                    bullish_score += 1
                    signal_details['volume_surge_up'] = "Volume surge on up day (bullish)"
                else:
                    bearish_score += 1
                    signal_details['volume_surge_down'] = "Volume surge on down day (bearish)"
            
            # Market context signals
            if market_context.get('market_trend') == 'bullish':
                bullish_score += 1
                signal_details['bullish_market'] = "Overall market is bullish"
            elif market_context.get('market_trend') == 'bearish':
                bearish_score += 1
                signal_details['bearish_market'] = "Overall market is bearish"
            
            if market_context.get('volatility') == 'high':
                signal_details['high_market_volatility'] = "Market volatility is high"
            elif market_context.get('volatility') == 'low':
                signal_details['low_market_volatility'] = "Market volatility is low"
            
            # Normalize scores
            total_score = bullish_score + bearish_score
            if total_score > 0:
                bullish = bullish_score / total_score
                bearish = bearish_score / total_score
            else:
                bullish = 0.5
                bearish = 0.5
            
            neutral = 1 - abs(bullish - bearish)
            
            return {
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral,
                'signal_details': signal_details
            }
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {'bullish': 0, 'bearish': 0, 'neutral': 1, 'signal_details': {}}
    
    def _filter_options(self, options_analysis, signals, strategy_types=None):
        """
        Filter options based on signals and strategy types
        
        Args:
            options_analysis (pd.DataFrame): Analyzed options data
            signals (dict): Trading signals
            strategy_types (list): Optional list of strategy types to consider
            
        Returns:
            pd.DataFrame: Filtered options
        """
        try:
            if options_analysis.empty:
                logger.warning("Cannot filter options: options analysis is empty")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            filtered_options = options_analysis.copy()
            
            # Filter by days to expiration (avoid options expiring too soon or too far out)
            filtered_options = filtered_options[
                (filtered_options['days_to_expiration'] >= 7) & 
                (filtered_options['days_to_expiration'] <= 60)
            ]
            
            if filtered_options.empty:
                logger.warning("No options meet expiration criteria")
                return pd.DataFrame()
            
            # Filter by liquidity
            filtered_options = filtered_options[filtered_options['liquidity_score'] > 0.2]
            
            if filtered_options.empty:
                logger.warning("No options meet liquidity criteria")
                return pd.DataFrame()
            
            # Filter by strategy type
            if not strategy_types:
                strategy_types = ['directional', 'income', 'volatility']
            
            strategy_filters = []
            
            if 'directional' in strategy_types:
                # Directional strategies based on signals
                if signals['bullish'] > 0.6:  # Strong bullish signal
                    # For bullish, consider calls
                    bullish_filter = (
                        (filtered_options['option_type'] == 'CALL') & 
                        (filtered_options['moneyness_category'].isin(['ATM', 'OTM']))
                    )
                    strategy_filters.append(bullish_filter)
                
                if signals['bearish'] > 0.6:  # Strong bearish signal
                    # For bearish, consider puts
                    bearish_filter = (
                        (filtered_options['option_type'] == 'PUT') & 
                        (filtered_options['moneyness_category'].isin(['ATM', 'OTM']))
                    )
                    strategy_filters.append(bearish_filter)
            
            if 'income' in strategy_types:
                # Income strategies (focus on probability of profit)
                income_filter = (filtered_options['probability_of_profit'] > 0.6)
                strategy_filters.append(income_filter)
            
            if 'volatility' in strategy_types:
                # Volatility strategies (focus on implied volatility)
                if signals.get('signal_details', {}).get('high_volatility'):
                    # High volatility environment
                    volatility_filter = (filtered_options['implied_volatility'] > 0.3)
                    strategy_filters.append(volatility_filter)
                elif signals.get('signal_details', {}).get('low_volatility'):
                    # Low volatility environment
                    volatility_filter = (filtered_options['implied_volatility'] < 0.3)
                    strategy_filters.append(volatility_filter)
            
            # Combine filters with OR logic
            if strategy_filters:
                combined_filter = strategy_filters[0]
                for filter_condition in strategy_filters[1:]:
                    combined_filter = combined_filter | filter_condition
                
                filtered_options = filtered_options[combined_filter]
            
            if filtered_options.empty:
                logger.warning("No options meet strategy criteria")
                return pd.DataFrame()
            
            return filtered_options
        
        except Exception as e:
            logger.error(f"Error filtering options: {str(e)}")
            return pd.DataFrame()
    
    def _score_options(self, filtered_options, signals, market_context):
        """
        Score and rank options based on multiple factors
        
        Args:
            filtered_options (pd.DataFrame): Filtered options data
            signals (dict): Trading signals
            market_context (dict): Market context data
            
        Returns:
            pd.DataFrame: Scored options
        """
        try:
            if filtered_options.empty:
                logger.warning("Cannot score options: filtered options is empty")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            scored_options = filtered_options.copy()
            
            # Initialize score components
            scored_options['signal_score'] = 0.0
            scored_options['risk_reward_score'] = 0.0
            scored_options['liquidity_score'] = scored_options['liquidity_score']  # Already calculated
            scored_options['time_score'] = 0.0
            
            # Calculate signal score based on option type and market signals
            scored_options['signal_score'] = scored_options.apply(
                lambda row: self._calculate_signal_score(row, signals),
                axis=1
            )
            
            # Calculate risk-reward score
            max_risk_reward = scored_options['risk_reward_ratio'].max()
            if max_risk_reward > 0:
                scored_options['risk_reward_score'] = scored_options['risk_reward_ratio'] / max_risk_reward
            
            # Calculate time score (prefer options with 30-45 days to expiration)
            scored_options['time_score'] = scored_options.apply(
                lambda row: 1.0 - abs(row['days_to_expiration'] - 40) / 40,
                axis=1
            )
            
            # Calculate final score (weighted average)
            scored_options['score'] = (
                scored_options['signal_score'] * 0.4 +
                scored_options['risk_reward_score'] * 0.3 +
                scored_options['liquidity_score'] * 0.2 +
                scored_options['time_score'] * 0.1
            )
            
            # Calculate confidence score
            scored_options['confidence'] = scored_options.apply(
                lambda row: self._calculate_confidence(row, signals, market_context),
                axis=1
            )
            
            # Sort by score (descending)
            scored_options = scored_options.sort_values('score', ascending=False)
            
            # Add recommendation details
            scored_options['recommendation_details'] = scored_options.apply(
                lambda row: self._create_recommendation_details(row, signals, market_context),
                axis=1
            )
            
            return scored_options
        
        except Exception as e:
            logger.error(f"Error scoring options: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_signal_score(self, option, signals):
        """
        Calculate signal score for an option based on market signals
        
        Args:
            option (pd.Series): Option data
            signals (dict): Trading signals
            
        Returns:
            float: Signal score (0-1)
        """
        try:
            if option['option_type'] == 'CALL':
                # For calls, higher score with bullish signals
                return signals['bullish']
            else:  # PUT
                # For puts, higher score with bearish signals
                return signals['bearish']
        except:
            return 0.5  # Default to neutral if calculation fails
    
    def _calculate_confidence(self, option, signals, market_context):
        """
        Calculate confidence score for an option recommendation
        
        Args:
            option (pd.Series): Option data
            signals (dict): Trading signals
            market_context (dict): Market context data
            
        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Base confidence on probability of profit
            base_confidence = option['probability_of_profit']
            
            # Adjust based on signal strength
            signal_strength = 0
            if option['option_type'] == 'CALL' and signals['bullish'] > 0.6:
                signal_strength = signals['bullish'] - 0.6
            elif option['option_type'] == 'PUT' and signals['bearish'] > 0.6:
                signal_strength = signals['bearish'] - 0.6
            
            # Adjust based on market context
            context_adjustment = 0
            if option['option_type'] == 'CALL' and market_context.get('market_trend') == 'bullish':
                context_adjustment = 0.1
            elif option['option_type'] == 'PUT' and market_context.get('market_trend') == 'bearish':
                context_adjustment = 0.1
            
            # Combine factors
            confidence = base_confidence + (signal_strength * 0.2) + context_adjustment
            
            # Ensure confidence is between 0 and 1
            return max(0, min(1, confidence))
        except:
            return 0.5  # Default to medium confidence if calculation fails
    
    def _create_recommendation_details(self, option, signals, market_context):
        """
        Create detailed recommendation information for an option
        
        Args:
            option (pd.Series): Option data
            signals (dict): Trading signals
            market_context (dict): Market context data
            
        Returns:
            dict: Recommendation details
        """
        try:
            details = {
                'type': option['option_type'],
                'symbol': option['symbol'],
                'strike': option['strike'],
                'expiration': option['expiration'],
                'days_to_expiration': int(option['days_to_expiration']),
                'current_price': option['mid_price'],
                'bid': option['bid'],
                'ask': option['ask'],
                'underlying_price': option['underlying_price'],
                'implied_volatility': option['implied_volatility'],
                'delta': option['delta'],
                'gamma': option['gamma'],
                'theta': option['theta'],
                'vega': option['vega'],
                'probability_of_profit': option['probability_of_profit'],
                'risk_reward_ratio': option['risk_reward_ratio'],
                'potential_return_pct': option['potential_return_pct'],
                'confidence_score': option['confidence'],
                'moneyness': option['moneyness'],
                'moneyness_category': option['moneyness_category'],
                'signal_details': signals.get('signal_details', {})
            }
            
            return details
        except Exception as e:
            logger.error(f"Error creating recommendation details: {str(e)}")
            return {}


class ShortTermRecommendationEngine(RecommendationEngine):
    """
    Specialized recommendation engine for short-term options trading strategies.
    Focuses on options with shorter expiration dates and more aggressive strategies.
    """
    
    def __init__(self, data_collector):
        """
        Initialize the short-term recommendation engine
        
        Args:
            data_collector: DataCollector instance for retrieving market data
        """
        super().__init__(data_collector)
        # Override cache settings for more frequent updates
        self.cache_expiry = 5 * 60  # 5 minutes in seconds
        # Configure logger
        self.logger = logging.getLogger('short_term_recommendation_engine')
        self.logger.setLevel(logging.INFO)
    
    def generate_recommendations(self, symbol, timeframe='30m', lookback_periods=20, confidence_threshold=0.5):
        """
        Generate short-term options trading recommendations
        
        Args:
            symbol (str): The stock symbol
            timeframe (str): Timeframe for analysis (e.g., '5m', '15m', '30m', '1h')
            lookback_periods (int): Number of periods to look back
            confidence_threshold (float): Minimum confidence score for recommendations
            
        Returns:
            list: Recommendations with details
        """
        try:
            self.logger.info(f"Generating short-term recommendations for {symbol} on {timeframe} timeframe")
            
            # Validate timeframe and convert to API parameters
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
            
            if timeframe not in valid_timeframes:
                self.logger.warning(f"Invalid timeframe: {timeframe}, defaulting to 30m")
                timeframe = '30m'
            
            # Get historical data for the specified timeframe
            historical_data = self._get_short_term_historical_data(symbol, timeframe, lookback_periods)
            
            if historical_data.empty:
                self.logger.warning(f"No historical data available for {symbol} on {timeframe} timeframe")
                return []
            
            # Get option chain
            option_chain = self.data_collector.get_option_chain_with_underlying_price(symbol)
            
            if not option_chain or 'options' not in option_chain or not option_chain['options']:
                self.logger.warning(f"No options data available for {symbol}")
                return []
            
            # Get current price
            current_price = option_chain.get('underlying_price', 0)
            if current_price <= 0:
                self.logger.warning(f"Invalid current price for {symbol}")
                # Try to get current price from another source
                quote_data = self.data_collector.get_quote(symbol)
                if quote_data:
                    # First try to use 'mark' from quote data
                    if 'mark' in quote_data and quote_data['mark'] > 0:
                        current_price = quote_data['mark']
                        self.logger.info(f"Using mark from quote for {symbol}: {current_price}")
                    # Then try to use 'lastPrice' from quote data
                    elif 'lastPrice' in quote_data and quote_data['lastPrice'] > 0:
                        current_price = quote_data['lastPrice']
                        self.logger.info(f"Using lastPrice from quote for {symbol}: {current_price}")
                    # Finally try to use 'underlyingPrice' from option chain
                    elif 'underlyingPrice' in option_chain and option_chain['underlyingPrice'] > 0:
                        current_price = option_chain['underlyingPrice']
                        self.logger.info(f"Using underlyingPrice from option chain for {symbol}: {current_price}")
            
            # Calculate short-term indicators
            indicators = self._calculate_short_term_indicators(historical_data)
            
            # Generate short-term signals
            signals = self._generate_short_term_signals(indicators)
            
            # Filter options for short-term strategies
            filtered_options = self._filter_short_term_options(option_chain['options'], current_price, signals)
            
            if not filtered_options:
                self.logger.warning(f"No suitable options found for {symbol} after filtering")
                return []
            
            # Score and rank options
            scored_options = self._score_short_term_options(filtered_options, signals, current_price)
            
            # Filter by confidence threshold
            recommendations = [opt for opt in scored_options if opt['confidence'] >= confidence_threshold]
            
            if not recommendations:
                self.logger.warning(f"No recommendations meet the confidence threshold of {confidence_threshold}")
                return []
            
            self.logger.info(f"Generated {len(recommendations)} short-term recommendations for {symbol}")
            return recommendations
        
        except Exception as e:
            self.logger.error(f"Error generating short-term recommendations: {str(e)}")
            return []
    
    def _get_short_term_historical_data(self, symbol, timeframe, lookback_periods):
        """
        Get short-term historical data for a symbol
        
        Args:
            symbol (str): The stock symbol
            timeframe (str): Timeframe for analysis
            lookback_periods (int): Number of periods to look back
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            # Map timeframe to API parameters
            timeframe_mapping = {
                '1m': ('day', 1, 'minute', 1),
                '5m': ('day', 1, 'minute', 5),
                '15m': ('day', 1, 'minute', 15),
                '30m': ('day', 1, 'minute', 30),
                '1h': ('day', 1, 'minute', 60),
                '2h': ('day', 2, 'minute', 120),
                '4h': ('day', 2, 'minute', 240),
                '1d': ('month', 1, 'daily', 1)
            }
            
            period_type, period, freq_type, freq = timeframe_mapping.get(timeframe, ('day', 1, 'minute', 30))
            
            # Cache key based on parameters
            cache_key = f"{symbol}_{timeframe}_data"
            
            # Check if we have cached data
            if hasattr(self, '_data_cache') and cache_key in self._data_cache:
                cache_time, cached_data = self._data_cache[cache_key]
                # Use cache if less than 5 minutes old
                if (time.time() - cache_time) < 300:
                    self.logger.info(f"Using cached data for {symbol} {timeframe}")
                    return cached_data
            
            self.logger.info(f"Fetching historical data for {symbol} with period_type={period_type}, period={period}, frequency_type={freq_type}, frequency={freq}")
            
            # Fetch historical data
            historical_data = self.data_collector.get_historical_data(
                symbol, 
                period_type=period_type, 
                period=period, 
                frequency_type=freq_type, 
                frequency=freq
            )
            
            if historical_data.empty:
                self.logger.warning(f"No historical data returned for {symbol}")
                return pd.DataFrame()
            
            # Cache the data
            if not hasattr(self, '_data_cache'):
                self._data_cache = {}
            self._data_cache[cache_key] = (time.time(), historical_data)
            
            self.logger.info(f"Fetched {len(historical_data)} historical data points for {symbol}")
            return historical_data
        
        except Exception as e:
            self.logger.error(f"Error fetching short-term historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_short_term_indicators(self, historical_data):
        """
        Calculate technical indicators for short-term analysis
        
        Args:
            historical_data (pd.DataFrame): Historical price data
            
        Returns:
            dict: Technical indicators
        """
        try:
            if historical_data.empty:
                self.logger.warning("Cannot calculate indicators: historical data is empty")
                return {}
            
            # Calculate short-term indicators
            indicators = {}
            
            # Short-term moving averages
            indicators['sma_5'] = self.technical_indicators.sma(historical_data['close'], window=5)
            indicators['sma_10'] = self.technical_indicators.sma(historical_data['close'], window=10)
            indicators['sma_20'] = self.technical_indicators.sma(historical_data['close'], window=20)
            indicators['ema_5'] = self.technical_indicators.ema(historical_data['close'], window=5)
            indicators['ema_10'] = self.technical_indicators.ema(historical_data['close'], window=10)
            
            # Momentum indicators
            indicators['rsi'] = self.technical_indicators.rsi(historical_data['close'], window=14)
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.technical_indicators.macd(
                historical_data['close'], fast_period=12, slow_period=26, signal_period=9
            )
            
            # Volatility indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.technical_indicators.bollinger_bands(
                historical_data['close'], window=20, num_std=2
            )
            
            # Get the most recent values for each indicator
            latest_indicators = {k: v.iloc[-1] if not pd.isna(v.iloc[-1]) else 0 for k, v in indicators.items()}
            
            # Add derived indicators
            latest_indicators['price'] = historical_data['close'].iloc[-1]
            latest_indicators['price_prev'] = historical_data['close'].iloc[-2] if len(historical_data) > 1 else latest_indicators['price']
            latest_indicators['volume'] = historical_data['volume'].iloc[-1]
            latest_indicators['volume_avg_5'] = historical_data['volume'].rolling(window=5).mean().iloc[-1]
            
            # Trend strength
            latest_indicators['above_sma_5'] = latest_indicators['price'] > latest_indicators['sma_5']
            latest_indicators['above_sma_10'] = latest_indicators['price'] > latest_indicators['sma_10']
            latest_indicators['above_sma_20'] = latest_indicators['price'] > latest_indicators['sma_20']
            
            # Short-term momentum
            latest_indicators['price_change_pct'] = (latest_indicators['price'] - latest_indicators['price_prev']) / latest_indicators['price_prev'] if latest_indicators['price_prev'] > 0 else 0
            
            return latest_indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating short-term indicators: {str(e)}")
            return {}
    
    def _generate_short_term_signals(self, indicators):
        """
        Generate trading signals for short-term strategies
        
        Args:
            indicators (dict): Technical indicators
            
        Returns:
            dict: Trading signals
        """
        try:
            if not indicators:
                self.logger.warning("Cannot generate signals: indicators dictionary is empty")
                return {'bullish': 0, 'bearish': 0, 'neutral': 1, 'signal_details': {}}
            
            # Initialize signal scores
            bullish_score = 0
            bearish_score = 0
            signal_details = {}
            
            # Short-term trend signals
            if indicators.get('above_sma_5', False):
                bullish_score += 1
                signal_details['above_sma_5'] = "Price is above 5-period SMA"
            else:
                bearish_score += 1
                signal_details['below_sma_5'] = "Price is below 5-period SMA"
            
            if indicators.get('above_sma_10', False):
                bullish_score += 1
                signal_details['above_sma_10'] = "Price is above 10-period SMA"
            else:
                bearish_score += 1
                signal_details['below_sma_10'] = "Price is below 10-period SMA"
            
            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                bearish_score += 2
                signal_details['rsi_overbought'] = f"RSI is overbought ({rsi:.1f})"
            elif rsi < 30:
                bullish_score += 2
                signal_details['rsi_oversold'] = f"RSI is oversold ({rsi:.1f})"
            elif rsi > 60:
                bullish_score += 1
                signal_details['rsi_bullish'] = f"RSI is bullish ({rsi:.1f})"
            elif rsi < 40:
                bearish_score += 1
                signal_details['rsi_bearish'] = f"RSI is bearish ({rsi:.1f})"
            
            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                bullish_score += 1
                signal_details['macd_above_signal'] = "MACD is above signal line"
            else:
                bearish_score += 1
                signal_details['macd_below_signal'] = "MACD is below signal line"
            
            if macd > 0:
                bullish_score += 1
                signal_details['macd_positive'] = "MACD is positive"
            else:
                bearish_score += 1
                signal_details['macd_negative'] = "MACD is negative"
            
            # Bollinger Band signals
            price = indicators.get('price', 0)
            bb_upper = indicators.get('bb_upper', price * 1.1)
            bb_lower = indicators.get('bb_lower', price * 0.9)
            bb_middle = indicators.get('bb_middle', price)
            
            if price > bb_upper:
                bearish_score += 1
                signal_details['above_bb_upper'] = "Price is above upper Bollinger Band (potential reversal)"
            elif price < bb_lower:
                bullish_score += 1
                signal_details['below_bb_lower'] = "Price is below lower Bollinger Band (potential reversal)"
            
            # Recent price movement
            price_change_pct = indicators.get('price_change_pct', 0)
            if price_change_pct > 0.01:  # 1% up
                bullish_score += 1
                signal_details['recent_upward_move'] = f"Recent price increase ({price_change_pct*100:.1f}%)"
            elif price_change_pct < -0.01:  # 1% down
                bearish_score += 1
                signal_details['recent_downward_move'] = f"Recent price decrease ({price_change_pct*100:.1f}%)"
            
            # Normalize scores
            total_score = bullish_score + bearish_score
            if total_score > 0:
                bullish = bullish_score / total_score
                bearish = bearish_score / total_score
            else:
                bullish = 0.5
                bearish = 0.5
            
            neutral = 1 - abs(bullish - bearish)
            
            return {
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral,
                'signal_details': signal_details
            }
        
        except Exception as e:
            self.logger.error(f"Error generating short-term signals: {str(e)}")
            return {'bullish': 0, 'bearish': 0, 'neutral': 1, 'signal_details': {}}
    
    def _filter_short_term_options(self, options, current_price, signals):
        """
        Filter options for short-term strategies
        
        Args:
            options (list): Options data
            current_price (float): Current price of the underlying
            signals (dict): Trading signals
            
        Returns:
            list: Filtered options
        """
        try:
            if not options:
                self.logger.warning("Cannot filter options: options list is empty")
                return []
            
            filtered_options = []
            
            # Convert expiration dates to datetime objects
            today = datetime.now().date()
            
            # Determine max expiration date (7 days for short-term)
            max_date = (datetime.now() + timedelta(days=7)).date()
            
            # Process each option
            for option in options:
                try:
                    # Parse expiration date
                    exp_date = datetime.strptime(option['expiration'], '%Y-%m-%d').date()
                    
                    # Log expiration date for debugging
                    self.logger.info(f"Expiration date: {exp_date}, Max date: {max_date}")
                    
                    # Skip options expiring too far in the future
                    if exp_date > max_date:
                        continue
                    
                    # Skip options with no volume or open interest
                    if option['volume'] < 10 and option['open_interest'] < 50:
                        continue
                    
                    # Calculate moneyness
                    strike = option['strike']
                    moneyness = strike / current_price if current_price > 0 else 1.0
                    
                    # Filter based on signals and option type
                    if signals['bullish'] > 0.6 and option['option_type'] == 'CALL':
                        # For bullish signals, consider calls near the money
                        if 0.95 <= moneyness <= 1.05:
                            filtered_options.append(option)
                    elif signals['bearish'] > 0.6 and option['option_type'] == 'PUT':
                        # For bearish signals, consider puts near the money
                        if 0.95 <= moneyness <= 1.05:
                            filtered_options.append(option)
                    elif signals['neutral'] > 0.6:
                        # For neutral signals, consider both calls and puts that are slightly OTM
                        if (option['option_type'] == 'CALL' and 1.0 <= moneyness <= 1.05) or \
                           (option['option_type'] == 'PUT' and 0.95 <= moneyness <= 1.0):
                            filtered_options.append(option)
                
                except Exception as e:
                    self.logger.error(f"Error processing option: {str(e)}")
                    continue
            
            self.logger.info(f"Filtered {len(filtered_options)} options for short-term strategies")
            return filtered_options
        
        except Exception as e:
            self.logger.error(f"Error filtering short-term options: {str(e)}")
            return []
    
    def _score_short_term_options(self, filtered_options, signals, current_price):
        """
        Score and rank options for short-term strategies
        
        Args:
            filtered_options (list): Filtered options data
            signals (dict): Trading signals
            current_price (float): Current price of the underlying
            
        Returns:
            list: Scored options
        """
        try:
            if not filtered_options:
                self.logger.warning("Cannot score options: filtered options list is empty")
                return []
            
            scored_options = []
            
            for option in filtered_options:
                try:
                    # Calculate base score based on option type and signals
                    if option['option_type'] == 'CALL':
                        base_score = signals['bullish']
                    else:  # PUT
                        base_score = signals['bearish']
                    
                    # Calculate liquidity score (0-1)
                    liquidity_score = min(1.0, (option['volume'] / 1000) * (1 / (1 + option['ask'] - option['bid'])))
                    
                    # Calculate moneyness score (higher for options closer to ATM)
                    strike = option['strike']
                    moneyness = strike / current_price if current_price > 0 else 1.0
                    moneyness_score = 1.0 - min(1.0, abs(moneyness - 1.0) * 10)
                    
                    # Calculate final score (weighted average)
                    final_score = (base_score * 0.5) + (liquidity_score * 0.3) + (moneyness_score * 0.2)
                    
                    # Calculate confidence score
                    confidence = base_score * 0.7 + liquidity_score * 0.3
                    
                    # Create recommendation details
                    recommendation = option.copy()
                    recommendation.update({
                        'score': final_score,
                        'confidence': confidence,
                        'signal_details': signals['signal_details']
                    })
                    
                    scored_options.append(recommendation)
                
                except Exception as e:
                    self.logger.error(f"Error scoring option: {str(e)}")
                    continue
            
            # Sort by score (descending)
            scored_options.sort(key=lambda x: x['score'], reverse=True)
            
            self.logger.info(f"Scored {len(scored_options)} options for short-term strategies")
            return scored_options
        
        except Exception as e:
            self.logger.error(f"Error scoring short-term options: {str(e)}")
            return []
