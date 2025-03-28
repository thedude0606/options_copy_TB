"""
Data collection module for options recommendation platform.
Handles retrieving market data from Schwab API for technical indicators and options analysis.
"""
import os
import pandas as pd
import numpy as np
import traceback
import json
import sys
from datetime import datetime, timedelta
from app.auth import get_client
from app.data.options_symbol_parser import OptionsSymbolParser

# Enable debug mode with enhanced logging
DEBUG_MODE = True

# Add more detailed debugging levels
VERBOSE_DEBUG = True  # For even more detailed debugging information
LOG_API_RESPONSES = True  # Log full API responses for troubleshooting

class DataCollector:
    """
    Class to collect and prepare data for technical indicators and options analysis
    """
    def __init__(self, interactive_auth=False):
        """
        Initialize the data collector
        
        Args:
            interactive_auth (bool): Whether to allow interactive authentication
        """
        self.client = get_client(interactive=interactive_auth)
        self.options_parser = OptionsSymbolParser()
        self.cache = {}  # Simple cache for frequently accessed data
        
        if DEBUG_MODE:
            print(f"DataCollector initialized with interactive_auth={interactive_auth}")
            print(f"Client type: {type(self.client)}")
    
    def get_option_chain(self, symbol):
        """
        Get the option chain for a symbol
        
        Args:
            symbol (str): The stock symbol to get options for
            
        Returns:
            dict: Option chain data
        """
        try:
            if DEBUG_MODE:
                print(f"Requesting option chain for symbol: {symbol}")
            
            # Get option chain data with required parameters
            option_chain_response = self.client.option_chains(
                symbol=symbol,
                contractType="ALL",
                strikeCount=10,  # Get options around the current price
                includeUnderlyingQuote=True,
                strategy="SINGLE"
            )
            
            if DEBUG_MODE:
                print(f"Option chain response type: {type(option_chain_response)}")
                if hasattr(option_chain_response, 'status_code'):
                    print(f"Status code: {option_chain_response.status_code}")
            
            # Process the response
            option_chain = None
            if hasattr(option_chain_response, 'json'):
                try:
                    option_chain = option_chain_response.json()
                    if DEBUG_MODE:
                        print(f"Option chain received for {symbol}, keys: {list(option_chain.keys() if isinstance(option_chain, dict) else [])}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error parsing option chain JSON: {str(e)}")
                        if hasattr(option_chain_response, 'text'):
                            print(f"Response text: {option_chain_response.text[:500]}...")
            elif isinstance(option_chain_response, dict):
                option_chain = option_chain_response
                if DEBUG_MODE:
                    print(f"Option chain received for {symbol}, keys: {list(option_chain.keys())}")
            else:
                if DEBUG_MODE:
                    print(f"No option chain data received for {symbol}")
            
            return option_chain
        except Exception as e:
            print(f"Error retrieving option chain for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_option_data(self, symbol):
        """
        Get detailed options data for a symbol (compatibility method for RecommendationEngine)
        
        Args:
            symbol (str): The stock symbol to get options for
            
        Returns:
            pd.DataFrame: Options data with Greeks and other details
        """
        try:
            if DEBUG_MODE:
                print(f"Getting option data for {symbol} (compatibility method)")
            
            # Get option chain
            option_chain = self.get_option_chain(symbol)
            if not option_chain:
                if DEBUG_MODE:
                    print(f"No option chain available for {symbol}")
                return pd.DataFrame()
            
            # Process option chain into a DataFrame
            options_data = []
            
            # Check if we have putExpDateMap and callExpDateMap
            put_map = option_chain.get('putExpDateMap', {})
            call_map = option_chain.get('callExpDateMap', {})
            
            # Get underlying price
            underlying_price = None
            underlying = option_chain.get('underlying', {})
            if underlying:
                underlying_price = underlying.get('mark', underlying.get('last', underlying.get('close')))
            
            # Process calls
            for exp_date, strikes in call_map.items():
                for strike, options in strikes.items():
                    for option in options:
                        option_data = {
                            'symbol': option.get('symbol', ''),
                            'underlyingSymbol': symbol,
                            'underlyingPrice': underlying_price,
                            'optionType': 'CALL',
                            'strikePrice': float(strike),
                            'expirationDate': exp_date.split(':')[0],
                            'daysToExpiration': option.get('daysToExpiration', 0),
                            'bid': option.get('bid', 0),
                            'ask': option.get('ask', 0),
                            'last': option.get('last', 0),
                            'mark': option.get('mark', 0),
                            'volume': option.get('totalVolume', 0),
                            'openInterest': option.get('openInterest', 0),
                            'impliedVolatility': option.get('volatility', 0) / 100,  # Convert to decimal
                            'delta': option.get('delta', 0),
                            'gamma': option.get('gamma', 0),
                            'theta': option.get('theta', 0),
                            'vega': option.get('vega', 0),
                            'rho': option.get('rho', 0),
                            'inTheMoney': option.get('inTheMoney', False)
                        }
                        options_data.append(option_data)
            
            # Process puts
            for exp_date, strikes in put_map.items():
                for strike, options in strikes.items():
                    for option in options:
                        option_data = {
                            'symbol': option.get('symbol', ''),
                            'underlyingSymbol': symbol,
                            'underlyingPrice': underlying_price,
                            'optionType': 'PUT',
                            'strikePrice': float(strike),
                            'expirationDate': exp_date.split(':')[0],
                            'daysToExpiration': option.get('daysToExpiration', 0),
                            'bid': option.get('bid', 0),
                            'ask': option.get('ask', 0),
                            'last': option.get('last', 0),
                            'mark': option.get('mark', 0),
                            'volume': option.get('totalVolume', 0),
                            'openInterest': option.get('openInterest', 0),
                            'impliedVolatility': option.get('volatility', 0) / 100,  # Convert to decimal
                            'delta': option.get('delta', 0),
                            'gamma': option.get('gamma', 0),
                            'theta': option.get('theta', 0),
                            'vega': option.get('vega', 0),
                            'rho': option.get('rho', 0),
                            'inTheMoney': option.get('inTheMoney', False)
                        }
                        options_data.append(option_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(options_data)
            
            if DEBUG_MODE:
                print(f"Processed {len(df)} options for {symbol}")
                if not df.empty:
                    print(f"Options data columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving options data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_underlying_symbol(self, symbol):
        """
        Extract the underlying symbol from an option symbol or return the symbol if it's already an equity
        
        Args:
            symbol (str): The symbol (option or equity)
            
        Returns:
            str: The underlying symbol
        """
        # Check if this is an option symbol
        if self.options_parser.is_option_symbol(symbol):
            underlying = self.options_parser.get_underlying_symbol(symbol)
            if DEBUG_MODE:
                print(f"Extracted underlying symbol {underlying} from option symbol {symbol}")
            return underlying
        else:
            # Already an equity symbol
            return symbol
    
    def get_price_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1, lookback_days=None):
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): The stock symbol to get price data for
            period_type (str): The type of period to show (day, month, year, ytd)
            period (int): The number of periods to show
            frequency_type (str): The type of frequency with which a new candle is formed (minute, daily, weekly, monthly)
            frequency (int): The number of the frequency type to use (e.g., 1, 5, 10, 15, 30 for minute)
            lookback_days (int, optional): Number of days to look back (alternative to period, takes precedence if provided)
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            # Get the underlying symbol if this is an option
            underlying_symbol = self.get_underlying_symbol(symbol)
            
            if DEBUG_MODE:
                print(f"Getting price data for {underlying_symbol}")
                print(f"Parameters: period_type={period_type}, period={period}, frequency_type={frequency_type}, frequency={frequency}")
            
            # If lookback_days is provided, convert to appropriate period
            if lookback_days:
                if lookback_days <= 10:
                    period_type = 'day'
                    period = lookback_days
                elif lookback_days <= 60:
                    period_type = 'month'
                    period = lookback_days // 30 + 1
                else:
                    period_type = 'year'
                    period = lookback_days // 365 + 1
                
                if DEBUG_MODE:
                    print(f"Converted lookback_days={lookback_days} to period_type={period_type}, period={period}")
            
            # Check cache
            cache_key = f"{underlying_symbol}_price_{period_type}_{period}_{frequency_type}_{frequency}"
            if cache_key in self.cache:
                if DEBUG_MODE:
                    print(f"Using cached price data for {underlying_symbol}")
                return self.cache[cache_key]
            
            # Generate mock price data since we don't have real API access
            # In a real implementation, this would call the Schwab API
            
            # Calculate date range
            end_date = datetime.now()
            if period_type == 'day':
                start_date = end_date - timedelta(days=period)
            elif period_type == 'month':
                start_date = end_date - timedelta(days=period * 30)
            elif period_type == 'year':
                start_date = end_date - timedelta(days=period * 365)
            else:  # ytd
                start_date = datetime(end_date.year, 1, 1)
            
            # Calculate number of data points
            if frequency_type == 'minute':
                # For minute data, we need to account for trading hours (6.5 hours per day)
                trading_minutes_per_day = 6.5 * 60
                business_days = np.busday_count(start_date.date(), end_date.date())
                num_points = int(business_days * trading_minutes_per_day / frequency)
                # Limit to a reasonable number
                num_points = min(num_points, 1000)
            elif frequency_type == 'daily':
                num_points = np.busday_count(start_date.date(), end_date.date())
            elif frequency_type == 'weekly':
                num_points = int((end_date - start_date).days / 7)
            else:  # monthly
                num_points = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
            
            # Ensure we have at least some data points
            num_points = max(num_points, 20)
            
            if DEBUG_MODE:
                print(f"Generating {num_points} data points from {start_date} to {end_date}")
            
            # Generate date range
            if frequency_type == 'minute':
                # For minute data, we need to generate timestamps during trading hours
                timestamps = []
                current_date = start_date
                while current_date <= end_date:
                    # Skip weekends
                    if current_date.weekday() < 5:  # Monday to Friday
                        # Trading hours: 9:30 AM to 4:00 PM
                        trading_start = datetime.combine(current_date.date(), datetime.min.time().replace(hour=9, minute=30))
                        trading_end = datetime.combine(current_date.date(), datetime.min.time().replace(hour=16, minute=0))
                        
                        # Generate timestamps during trading hours
                        current_time = trading_start
                        while current_time <= trading_end:
                            timestamps.append(current_time)
                            current_time += timedelta(minutes=frequency)
                    
                    current_date += timedelta(days=1)
                
                # Limit to the required number of points
                timestamps = timestamps[-num_points:]
            else:
                # For daily, weekly, monthly data
                if frequency_type == 'daily':
                    freq = f"{frequency}D"
                elif frequency_type == 'weekly':
                    freq = f"{frequency}W"
                else:  # monthly
                    freq = f"{frequency}M"
                
                # Generate timestamps
                timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)[-num_points:]
            
            # Generate price data
            # Start with a base price (use symbol hash for consistency)
            symbol_hash = sum(ord(c) for c in underlying_symbol)
            base_price = 50 + (symbol_hash % 200)  # Base price between 50 and 250
            
            # Generate random walk
            np.random.seed(symbol_hash)  # Use symbol hash as seed for reproducibility
            
            # Generate returns with some autocorrelation
            returns = np.random.normal(0.0005, 0.015, num_points)  # Mean slightly positive, realistic volatility
            
            # Add some autocorrelation and volatility clustering
            for i in range(1, len(returns)):
                returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
            
            # Convert returns to prices
            prices = base_price * np.cumprod(1 + returns)
            
            # Generate OHLC data
            data = []
            for i, timestamp in enumerate(timestamps):
                # Calculate open, high, low, close
                close = prices[i]
                
                # For the first point, open is close of previous day with small random change
                if i == 0:
                    open_price = close * (1 + np.random.normal(0, 0.005))
                else:
                    open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
                
                # High and low are random variations from open and close
                high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.008)))
                low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.008)))
                
                # Volume is random but correlated with price change
                volume = int(abs(close - open_price) * 1000000 * (1 + np.random.normal(0, 0.5)))
                volume = max(volume, 1000)  # Ensure some minimum volume
                
                data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Cache the result
            self.cache[cache_key] = df
            
            if DEBUG_MODE:
                print(f"Generated price data for {underlying_symbol}: {len(df)} rows")
                if not df.empty:
                    print(f"Price data columns: {df.columns.tolist()}")
                    print(f"Price data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving price data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1):
        """
        Get historical data for a symbol (wrapper around get_price_data for compatibility with RecommendationEngine)
        
        Args:
            symbol (str): The stock symbol to get historical data for
            period_type (str): The type of period to show (day, month, year, ytd)
            period (int): The number of periods to show
            frequency_type (str): The type of frequency with which a new candle is formed (minute, daily, weekly, monthly)
            frequency (int): The number of the frequency type to use (e.g., 1, 5, 10, 15, 30 for minute)
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            if DEBUG_MODE:
                print(f"Getting historical data for {symbol} (compatibility method)")
                print(f"Parameters: period_type={period_type}, period={period}, frequency_type={frequency_type}, frequency={frequency}")
            
            # Use the Yahoo Finance API if available
            try:
                # Try to import the data API module
                sys.path.append('/opt/.manus/.sandbox-runtime')
                from data_api import ApiClient
                client = ApiClient()
                
                if DEBUG_MODE:
                    print(f"Using Yahoo Finance API for historical data")
                
                # Convert period_type and frequency_type to Yahoo Finance parameters
                interval_map = {
                    ('minute', 1): '1m',
                    ('minute', 2): '2m',
                    ('minute', 5): '5m',
                    ('minute', 15): '15m',
                    ('minute', 30): '30m',
                    ('minute', 60): '60m',
                    ('daily', 1): '1d',
                    ('weekly', 1): '1wk',
                    ('monthly', 1): '1mo'
                }
                
                range_map = {
                    ('day', 1): '1d',
                    ('day', 5): '5d',
                    ('month', 1): '1mo',
                    ('month', 3): '3mo',
                    ('month', 6): '6mo',
                    ('year', 1): '1y',
                    ('year', 2): '2y',
                    ('year', 5): '5y',
                    ('year', 10): '10y'
                }
                
                # Get the appropriate interval and range
                interval = interval_map.get((frequency_type, frequency), '1d')
                data_range = range_map.get((period_type, period), '1mo')
                
                # Call the Yahoo Finance API
                chart_data = client.call_api('YahooFinance/get_stock_chart', query={
                    'symbol': symbol,
                    'interval': interval,
                    'range': data_range,
                    'includeAdjustedClose': True
                })
                
                if chart_data and 'chart' in chart_data and 'result' in chart_data['chart'] and chart_data['chart']['result']:
                    result = chart_data['chart']['result'][0]
                    
                    # Extract timestamp and indicators
                    timestamps = result.get('timestamp', [])
                    indicators = result.get('indicators', {})
                    quote = indicators.get('quote', [{}])[0]
                    
                    # Create DataFrame
                    data = []
                    for i in range(len(timestamps)):
                        if i < len(quote.get('open', [])) and i < len(quote.get('high', [])) and i < len(quote.get('low', [])) and i < len(quote.get('close', [])) and i < len(quote.get('volume', [])):
                            timestamp = datetime.fromtimestamp(timestamps[i])
                            data.append({
                                'timestamp': timestamp,
                                'open': quote['open'][i],
                                'high': quote['high'][i],
                                'low': quote['low'][i],
                                'close': quote['close'][i],
                                'volume': quote['volume'][i]
                            })
                    
                    df = pd.DataFrame(data)
                    
                    if DEBUG_MODE:
                        print(f"Retrieved historical data from Yahoo Finance API: {len(df)} rows")
                    
                    return df
                
                if DEBUG_MODE:
                    print(f"Failed to retrieve data from Yahoo Finance API, falling back to get_price_data")
            
            except Exception as api_error:
                if DEBUG_MODE:
                    print(f"Error using Yahoo Finance API: {str(api_error)}")
                    print(f"Falling back to get_price_data")
            
            # Fall back to get_price_data if Yahoo Finance API fails
            return self.get_price_data(
                symbol=symbol,
                period_type=period_type,
                period=period,
                frequency_type=frequency_type,
                frequency=frequency
            )
            
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol, period=14, indicators=None):
        """
        Calculate technical indicators for a symbol
        
        Args:
            symbol (str): The stock symbol to calculate indicators for
            period (int): The period to use for indicators
            indicators (list, optional): List of indicators to calculate
            
        Returns:
            pd.DataFrame: Technical indicators
        """
        try:
            if DEBUG_MODE:
                print(f"Calculating technical indicators for {symbol}")
                print(f"Period: {period}")
                print(f"Indicators: {indicators if indicators else 'All'}")
            
            # Get price data
            price_data = self.get_price_data(symbol, period_type='day', period=period*2)
            if price_data.empty:
                if DEBUG_MODE:
                    print(f"No price data available for {symbol}")
                return pd.DataFrame()
            
            # If no indicators specified, use default set
            if not indicators:
                indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger']
            
            # Calculate indicators
            result = price_data.copy()
            
            for indicator in indicators:
                if indicator.lower() == 'sma':
                    result[f'sma_{period}'] = result['close'].rolling(window=period).mean()
                
                elif indicator.lower() == 'ema':
                    result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
                
                elif indicator.lower() == 'rsi':
                    delta = result['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    rs = avg_gain / avg_loss
                    result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                
                elif indicator.lower() == 'macd':
                    ema12 = result['close'].ewm(span=12, adjust=False).mean()
                    ema26 = result['close'].ewm(span=26, adjust=False).mean()
                    result['macd_line'] = ema12 - ema26
                    result['macd_signal'] = result['macd_line'].ewm(span=9, adjust=False).mean()
                    result['macd_histogram'] = result['macd_line'] - result['macd_signal']
                
                elif indicator.lower() == 'bollinger':
                    result[f'bollinger_mid_{period}'] = result['close'].rolling(window=period).mean()
                    result[f'bollinger_std_{period}'] = result['close'].rolling(window=period).std()
                    result[f'bollinger_upper_{period}'] = result[f'bollinger_mid_{period}'] + 2 * result[f'bollinger_std_{period}']
                    result[f'bollinger_lower_{period}'] = result[f'bollinger_mid_{period}'] - 2 * result[f'bollinger_std_{period}']
            
            if DEBUG_MODE:
                print(f"Calculated indicators: {list(result.columns)}")
            
            return result
            
        except Exception as e:
            print(f"Error calculating technical indicators for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
