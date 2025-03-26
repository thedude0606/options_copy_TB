"""
Enhanced data pipeline for short-term options trading.
Optimized for 15, 30, 60, and 120-minute timeframes.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import re

class ShortTermDataPipeline:
    """
    Data pipeline optimized for short-term options trading timeframes
    """
    def __init__(self, data_collector):
        """
        Initialize the short-term data pipeline
        
        Args:
            data_collector: DataCollector instance for retrieving market data
        """
        self.data_collector = data_collector
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_expiry = {
            '1m': 60,  # 1 minute data expires after 60 seconds
            '5m': 300,  # 5 minute data expires after 5 minutes
            '15m': 900,  # 15 minute data expires after 15 minutes
            '30m': 1800,  # 30 minute data expires after 30 minutes
            '60m': 3600,  # 60 minute data expires after 1 hour
            '120m': 7200  # 120 minute data expires after 2 hours
        }
        self.logger = logging.getLogger('short_term_data_pipeline')
        
    def standardize_timeframe(self, timeframe):
        """
        Standardize timeframe format to ensure consistency
        
        Args:
            timeframe (str): Timeframe in various formats (e.g., '30m', '30min', '1h', '60min')
            
        Returns:
            str: Standardized timeframe format ('15m', '30m', '60m', '120m')
        """
        # If already in standard format, return as is
        if timeframe in ['15m', '30m', '60m', '120m']:
            return timeframe
            
        # Convert to lowercase and remove any whitespace
        tf = timeframe.lower().strip()
        
        # Extract numeric value and unit using regex
        match = re.match(r'(\d+)\s*([a-z]+)', tf)
        if not match:
            self.logger.warning(f"Could not parse timeframe format: {timeframe}, defaulting to 30m")
            return '30m'
            
        value, unit = match.groups()
        value = int(value)
        
        # Convert to standard format
        if unit in ['m', 'min', 'minute', 'minutes']:
            if value == 15:
                return '15m'
            elif value == 30:
                return '30m'
            elif value in [60, 1] and unit.startswith('m'):  # 60min or 1min
                return '60m' if value == 60 else '1m'
            elif value == 120:
                return '120m'
        elif unit in ['h', 'hr', 'hour', 'hours']:
            if value == 1:
                return '60m'
            elif value == 2:
                return '120m'
                
        # If no match found, default to closest standard timeframe
        if value < 22:  # Closer to 15m
            return '15m'
        elif value < 45:  # Closer to 30m
            return '30m'
        elif value < 90:  # Closer to 60m
            return '60m'
        else:  # Closer to 120m
            return '120m'
        
    def get_short_term_price_data(self, symbol, timeframe='30m', force_refresh=False):
        """
        Get price data for short-term analysis
        
        Args:
            symbol (str): Stock symbol
            timeframe (str): Timeframe for data ('15m', '30m', '60m', '120m')
            force_refresh (bool): Force refresh of cached data
            
        Returns:
            pd.DataFrame: Price data with OHLCV columns
        """
        # Standardize timeframe format
        timeframe = self.standardize_timeframe(timeframe)
        
        cache_key = f"{symbol}_{timeframe}"
        current_time = time.time()
        
        # Check if we have cached data that's still valid
        if not force_refresh and cache_key in self.cache:
            last_update = self.cache_timestamps.get(cache_key, 0)
            if current_time - last_update < self.cache_expiry.get(timeframe, 300):
                self.logger.info(f"Using cached data for {symbol} {timeframe}")
                return self.cache[cache_key]
        
        # Map timeframe to appropriate period type and frequency
        period_mapping = {
            '15m': ('day', 'minute', 15, 1),  # 1 day of 15-minute candles
            '30m': ('day', 'minute', 30, 1),  # 1 day of 30-minute candles
            '60m': ('day', 'minute', 60, 2),  # 2 days of 60-minute candles
            '120m': ('day', 'minute', 120, 3)  # 3 days of 120-minute candles
        }
        
        if timeframe not in period_mapping:
            self.logger.warning(f"Invalid timeframe: {timeframe}, defaulting to 30m")
            timeframe = '30m'
            
        period_type, frequency_type, frequency, period = period_mapping[timeframe]
        
        try:
            # Get historical price data
            price_data = self.data_collector.get_price_history(
                symbol=symbol,
                period_type=period_type,
                period=period,
                frequency_type=frequency_type,
                frequency=frequency
            )
            
            # Process the data
            if price_data is not None and not price_data.empty:
                # Ensure we have all required columns
                required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in price_data.columns:
                        self.logger.error(f"Missing required column: {col}")
                        return pd.DataFrame(columns=required_columns)
                
                # Convert datetime to proper format if it's not already
                if not pd.api.types.is_datetime64_any_dtype(price_data['datetime']):
                    price_data['datetime'] = pd.to_datetime(price_data['datetime'], unit='ms')
                
                # Sort by datetime
                price_data = price_data.sort_values('datetime')
                
                # Cache the data
                self.cache[cache_key] = price_data
                self.cache_timestamps[cache_key] = current_time
                
                return price_data
            else:
                self.logger.warning(f"No price data returned for {symbol} {timeframe}")
                return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                
        except Exception as e:
            self.logger.error(f"Error retrieving price data: {str(e)}")
            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_options_data_for_timeframe(self, symbol, timeframe='30m'):
        """
        Get options data optimized for the specified short-term timeframe
        
        Args:
            symbol (str): Stock symbol
            timeframe (str): Timeframe for trading ('15m', '30m', '60m', '120m')
            
        Returns:
            dict: Options data optimized for the timeframe
        """
        # Standardize timeframe format
        timeframe = self.standardize_timeframe(timeframe)
        
        # Map timeframe to appropriate expiration filter
        expiration_filters = {
            '15m': 1,  # Include options expiring today and tomorrow
            '30m': 2,  # Include options expiring within 2 days
            '60m': 3,  # Include options expiring within 3 days
            '120m': 5  # Include options expiring within 5 days
        }
        
        days_to_include = expiration_filters.get(timeframe, 2)
        
        try:
            # Get options chain
            options_chain = self.data_collector.get_options_chain(symbol)
            
            # Check if we have a valid options chain with the expected structure
            if not options_chain:
                self.logger.warning(f"No options data returned for {symbol}")
                return {'options': [], 'expirations': []}
                
            # Check for the actual structure returned by Schwab API
            # Based on logs, we expect callExpDateMap and putExpDateMap instead of 'options'
            if 'callExpDateMap' not in options_chain and 'putExpDateMap' not in options_chain:
                self.logger.warning(f"No options data returned for {symbol}")
                return {'options': [], 'expirations': []}
            
            # Extract underlying price
            underlying_price = options_chain.get('underlyingPrice', 0)
            
            # Process call and put options
            filtered_options = []
            expirations = set()
            
            # Add debug logging
            self.logger.info(f"Processing options chain for {symbol}")
            self.logger.info(f"Call expiration dates: {list(options_chain.get('callExpDateMap', {}).keys())}")
            self.logger.info(f"Put expiration dates: {list(options_chain.get('putExpDateMap', {}).keys())}")
            
            # Process call options
            call_exp_map = options_chain.get('callExpDateMap', {})
            for exp_date_str, strikes in call_exp_map.items():
                # Extract date part (format: "YYYY-MM-DD:X")
                exp_date_parts = exp_date_str.split(':')
                if len(exp_date_parts) > 0:
                    exp_date_str = exp_date_parts[0]
                    
                try:
                    exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                    # Check if this expiration is within our timeframe filter
                    today = datetime.now().date()
                    max_date = today + timedelta(days=days_to_include)
                    
                    # Add debug logging
                    self.logger.info(f"Expiration date: {exp_date}, Max date: {max_date}")
                    
                    # Include this expiration regardless of the filter to ensure we have some options
                    expirations.add(exp_date_str)
                    
                    # Process each strike price
                    for strike_str, strike_data in strikes.items():
                        for option in strike_data:
                            # Create a standardized option object
                            option_obj = {
                                'symbol': option.get('symbol', ''),
                                'expiration': exp_date_str,
                                'strike': float(strike_str),
                                'option_type': 'CALL',
                                'bid': option.get('bid', 0),
                                'ask': option.get('ask', 0),
                                'last': option.get('last', 0),
                                'mark': option.get('mark', 0),
                                'delta': option.get('delta', 0),
                                'gamma': option.get('gamma', 0),
                                'theta': option.get('theta', 0),
                                'vega': option.get('vega', 0),
                                'rho': option.get('rho', 0),
                                'volume': option.get('totalVolume', 0),
                                'open_interest': option.get('openInterest', 0),
                                'implied_volatility': option.get('volatility', 0) / 100 if option.get('volatility') else 0,
                                'in_the_money': option.get('inTheMoney', False)
                            }
                            filtered_options.append(option_obj)
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error processing call expiration date: {str(e)}")
                    continue
            
            # Process put options
            put_exp_map = options_chain.get('putExpDateMap', {})
            for exp_date_str, strikes in put_exp_map.items():
                # Extract date part (format: "YYYY-MM-DD:X")
                exp_date_parts = exp_date_str.split(':')
                if len(exp_date_parts) > 0:
                    exp_date_str = exp_date_parts[0]
                    
                try:
                    exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                    # Check if this expiration is within our timeframe filter
                    today = datetime.now().date()
                    max_date = today + timedelta(days=days_to_include)
                    
                    # Include this expiration regardless of the filter to ensure we have some options
                    expirations.add(exp_date_str)
                    
                    # Process each strike price
                    for strike_str, strike_data in strikes.items():
                        for option in strike_data:
                            # Create a standardized option object
                            option_obj = {
                                'symbol': option.get('symbol', ''),
                                'expiration': exp_date_str,
                                'strike': float(strike_str),
                                'option_type': 'PUT',
                                'bid': option.get('bid', 0),
                                'ask': option.get('ask', 0),
                                'last': option.get('last', 0),
                                'mark': option.get('mark', 0),
                                'delta': option.get('delta', 0),
                                'gamma': option.get('gamma', 0),
                                'theta': option.get('theta', 0),
                                'vega': option.get('vega', 0),
                                'rho': option.get('rho', 0),
                                'volume': option.get('totalVolume', 0),
                                'open_interest': option.get('openInterest', 0),
                                'implied_volatility': option.get('volatility', 0) / 100 if option.get('volatility') else 0,
                                'in_the_money': option.get('inTheMoney', False)
                            }
                            filtered_options.append(option_obj)
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error processing put expiration date: {str(e)}")
                    continue
            
            # Add debug logging for the number of options found
            self.logger.info(f"Found {len(filtered_options)} options for {symbol}")
            
            # If no options were found, include all options regardless of expiration date
            if not filtered_options:
                self.logger.warning(f"No options found within {days_to_include} days, including all available options")
                
                # Process all call options without date filtering
                for exp_date_str, strikes in call_exp_map.items():
                    exp_date_parts = exp_date_str.split(':')
                    if len(exp_date_parts) > 0:
                        exp_date_str = exp_date_parts[0]
                        expirations.add(exp_date_str)
                    
                    for strike_str, strike_data in strikes.items():
                        for option in strike_data:
                            option_obj = {
                                'symbol': option.get('symbol', ''),
                                'expiration': exp_date_str,
                                'strike': float(strike_str),
                                'option_type': 'CALL',
                                'bid': option.get('bid', 0),
                                'ask': option.get('ask', 0),
                                'last': option.get('last', 0),
                                'mark': option.get('mark', 0),
                                'delta': option.get('delta', 0),
                                'gamma': option.get('gamma', 0),
                                'theta': option.get('theta', 0),
                                'vega': option.get('vega', 0),
                                'rho': option.get('rho', 0),
                                'volume': option.get('totalVolume', 0),
                                'open_interest': option.get('openInterest', 0),
                                'implied_volatility': option.get('volatility', 0) / 100 if option.get('volatility') else 0,
                                'in_the_money': option.get('inTheMoney', False)
                            }
                            filtered_options.append(option_obj)
                
                # Process all put options without date filtering
                for exp_date_str, strikes in put_exp_map.items():
                    exp_date_parts = exp_date_str.split(':')
                    if len(exp_date_parts) > 0:
                        exp_date_str = exp_date_parts[0]
                        expirations.add(exp_date_str)
                    
                    for strike_str, strike_data in strikes.items():
                        for option in strike_data:
                            option_obj = {
                                'symbol': option.get('symbol', ''),
                                'expiration': exp_date_str,
                                'strike': float(strike_str),
                                'option_type': 'PUT',
                                'bid': option.get('bid', 0),
                                'ask': option.get('ask', 0),
                                'last': option.get('last', 0),
                                'mark': option.get('mark', 0),
                                'delta': option.get('delta', 0),
                                'gamma': option.get('gamma', 0),
                                'theta': option.get('theta', 0),
                                'vega': option.get('vega', 0),
                                'rho': option.get('rho', 0),
                                'volume': option.get('totalVolume', 0),
                                'open_interest': option.get('openInterest', 0),
                                'implied_volatility': option.get('volatility', 0) / 100 if option.get('volatility') else 0,
                                'in_the_money': option.get('inTheMoney', False)
                            }
                            filtered_options.append(option_obj)
                
                self.logger.info(f"After including all options, found {len(filtered_options)} options for {symbol}")
            
            result = {
                'symbol': symbol,
                'underlying_price': underlying_price,
                'options': filtered_options,
                'expirations': sorted(list(expirations))
            }
            
            # Final check to ensure we're returning options
            if not result['options']:
                self.logger.warning(f"Still no options found for {symbol} after processing")
            else:
                self.logger.info(f"Successfully processed {len(result['options'])} options for {symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving options data: {str(e)}")
            return {'options': [], 'expirations': []}
    
    def get_real_time_data(self, symbol):
        """
        Get real-time data for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Real-time market data
        """
        try:
            # Get real-time quote
            quote_data = self.data_collector.get_quote(symbol)
            
            if quote_data is None:
                self.logger.warning(f"No quote data returned for {symbol}")
                return {}
            
            # Format the data
            real_time_data = {
                'symbol': symbol,
                'last_price': quote_data.get('lastPrice', 0),
                'change': quote_data.get('netChange', 0),
                'change_percent': quote_data.get('netPercentChangeInDouble', 0),
                'bid': quote_data.get('bidPrice', 0),
                'ask': quote_data.get('askPrice', 0),
                'volume': quote_data.get('totalVolume', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            return real_time_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving real-time data: {str(e)}")
            return {}
    
    def get_market_overview(self):
        """
        Get market overview data for the dashboard
        
        Returns:
            dict: Market overview data
        """
        # List of major indices to track
        indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']
        
        market_data = {}
        for index in indices:
            try:
                quote = self.data_collector.get_quote(index)
                if quote:
                    market_data[index] = {
                        'last_price': quote.get('lastPrice', 0),
                        'change': quote.get('netChange', 0),
                        'change_percent': quote.get('netPercentChangeInDouble', 0)
                    }
            except Exception as e:
                self.logger.warning(f"Error retrieving data for {index}: {str(e)}")
        
        return market_data
    
    def get_sentiment_data(self, symbol):
        """
        Get sentiment data for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Sentiment data
        """
        # This is a placeholder for sentiment data integration
        # In a real implementation, this would connect to news and social media APIs
        
        # Simulate sentiment data for now
        sentiment_score = np.random.normal(0, 1)  # Random score with mean 0, std 1
        
        sentiment_data = {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'sentiment_label': 'Bullish' if sentiment_score > 0.5 else 'Bearish' if sentiment_score < -0.5 else 'Neutral',
            'news_count': int(np.random.randint(0, 10)),
            'social_mentions': int(np.random.randint(0, 100)),
            'timestamp': datetime.now().isoformat()
        }
        
        return sentiment_data
    
    def clear_cache(self, symbol=None, timeframe=None):
        """
        Clear the data cache
        
        Args:
            symbol (str, optional): Clear cache for specific symbol
            timeframe (str, optional): Clear cache for specific timeframe
        """
        if symbol and timeframe:
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.cache:
                del self.cache[cache_key]
                if cache_key in self.cache_timestamps:
                    del self.cache_timestamps[cache_key]
        elif symbol:
            keys_to_delete = [k for k in self.cache.keys() if k.startswith(f"{symbol}_")]
            for key in keys_to_delete:
                del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
        elif timeframe:
            keys_to_delete = [k for k in self.cache.keys() if k.endswith(f"_{timeframe}")]
            for key in keys_to_delete:
                del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
