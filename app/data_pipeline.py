"""
Enhanced data pipeline for short-term options trading.
Optimized for 15, 30, 60, and 120-minute timeframes.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

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
        # Map timeframe to appropriate expiration filter
        expiration_filters = {
            '15m': 0,  # Same day expiration
            '30m': 0,  # Same day expiration
            '60m': 1,  # Include tomorrow's expiration
            '120m': 2  # Include expiration up to 2 days out
        }
        
        days_to_include = expiration_filters.get(timeframe, 0)
        
        try:
            # Get options chain
            options_chain = self.data_collector.get_options_chain(symbol)
            
            if not options_chain or 'options' not in options_chain:
                self.logger.warning(f"No options data returned for {symbol}")
                return {'options': [], 'expirations': []}
            
            # Filter expirations based on timeframe
            today = datetime.now().date()
            max_date = today + timedelta(days=days_to_include)
            
            filtered_options = []
            for option in options_chain.get('options', []):
                try:
                    exp_date = datetime.strptime(option['expiration'], '%Y-%m-%d').date()
                    if exp_date <= max_date:
                        filtered_options.append(option)
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error processing expiration date: {str(e)}")
                    continue
            
            # Get unique expirations from filtered options
            expirations = list(set([opt['expiration'] for opt in filtered_options if 'expiration' in opt]))
            
            return {
                'symbol': symbol,
                'underlying_price': options_chain.get('underlying_price', 0),
                'options': filtered_options,
                'expirations': sorted(expirations)
            }
            
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
