"""
Data collection module for options recommendation platform.
Handles retrieving market data from Schwab API for technical indicators and options analysis.
"""
import os
import pandas as pd
import numpy as np
import traceback
import json
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
            
            # If lookback_days is provided, calculate the start and end dates
            if lookback_days is not None:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                
                if DEBUG_MODE:
                    print(f"Using lookback_days={lookback_days}, start_date={start_date}, end_date={end_date}")
                
                # Generate mock data for testing
                # In a real implementation, this would call the API with start_date and end_date
                return self._generate_mock_price_data(underlying_symbol, start_date, end_date, frequency_type, frequency)
            
            # For now, generate mock data for testing
            # In a real implementation, this would call the API with period_type, period, etc.
            end_date = datetime.now()
            if period_type == 'day':
                start_date = end_date - timedelta(days=period)
            elif period_type == 'month':
                start_date = end_date - timedelta(days=period * 30)
            elif period_type == 'year':
                start_date = end_date - timedelta(days=period * 365)
            else:  # ytd
                start_date = datetime(end_date.year, 1, 1)
            
            if DEBUG_MODE:
                print(f"Generating {period} {period_type}s of data from {start_date} to {end_date}")
            
            return self._generate_mock_price_data(underlying_symbol, start_date, end_date, frequency_type, frequency)
            
        except Exception as e:
            print(f"Error retrieving price data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _generate_mock_price_data(self, symbol, start_date, end_date, frequency_type, frequency):
        """
        Generate mock price data for testing
        
        Args:
            symbol (str): The stock symbol
            start_date (datetime): Start date for the data
            end_date (datetime): End date for the data
            frequency_type (str): The type of frequency (minute, daily, weekly, monthly)
            frequency (int): The frequency value
            
        Returns:
            pd.DataFrame: Mock price data
        """
        if DEBUG_MODE:
            print(f"Generating mock price data for {symbol}")
            print(f"Parameters: start_date={start_date}, end_date={end_date}, frequency_type={frequency_type}, frequency={frequency}")
        
        # Determine the number of data points based on frequency
        if frequency_type == 'minute':
            # Calculate business minutes between start and end
            # Assuming 6.5 hours of trading per day (390 minutes) and 5 trading days per week
            business_days = np.busday_count(
                np.datetime64(start_date.date()),
                np.datetime64(end_date.date())
            )
            minutes_per_day = 390  # 6.5 hours * 60 minutes
            total_minutes = business_days * minutes_per_day
            num_points = total_minutes // frequency
            
            # Limit to a reasonable number for testing
            num_points = min(num_points, 1000)
        elif frequency_type == 'daily':
            business_days = np.busday_count(
                np.datetime64(start_date.date()),
                np.datetime64(end_date.date())
            )
            num_points = business_days // frequency
        elif frequency_type == 'weekly':
            weeks = (end_date - start_date).days // 7
            num_points = weeks // frequency
        else:  # monthly
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            num_points = months // frequency
        
        # Ensure we have at least one data point
        num_points = max(num_points, 1)
        
        if DEBUG_MODE:
            print(f"Generating {num_points} data points")
        
        # Generate timestamps
        if frequency_type == 'minute':
            # Generate business day timestamps during trading hours
            timestamps = []
            current_date = start_date.date()
            while current_date <= end_date.date() and len(timestamps) < num_points:
                # Skip weekends
                if current_date.weekday() < 5:  # Monday to Friday
                    # Trading hours: 9:30 AM to 4:00 PM
                    for hour in range(9, 16):
                        for minute in range(0, 60, frequency):
                            if hour == 9 and minute < 30:
                                continue  # Skip before 9:30 AM
                            if hour == 16 and minute > 0:
                                continue  # Skip after 4:00 PM
                            
                            current_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
                            if current_time <= end_date and len(timestamps) < num_points:
                                timestamps.append(current_time)
                
                current_date += timedelta(days=1)
        else:
            # For daily, weekly, monthly, generate evenly spaced timestamps
            timestamps = [start_date + (end_date - start_date) * i / num_points for i in range(num_points)]
        
        # Generate price data
        base_price = 100.0  # Starting price
        volatility = 0.02  # Daily volatility
        
        # Adjust volatility based on frequency
        if frequency_type == 'minute':
            volatility = volatility / np.sqrt(390)  # Scale by sqrt of minutes in a trading day
        elif frequency_type == 'weekly':
            volatility = volatility * np.sqrt(5)  # Scale by sqrt of days in a week
        elif frequency_type == 'monthly':
            volatility = volatility * np.sqrt(21)  # Scale by sqrt of trading days in a month
        
        # Generate random walk
        np.random.seed(hash(symbol) % 2**32)  # Seed based on symbol for consistency
        returns = np.random.normal(0.0005, volatility, num_points)  # Small positive drift
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            high_low_range = price * volatility * 2
            
            # Create a row with timestamp and OHLCV data
            row = {
                'timestamp': timestamp,
                'open': price * (1 - volatility / 2),
                'high': price + high_low_range / 2,
                'low': price - high_low_range / 2,
                'close': price,
                'volume': np.random.randint(1000, 1000000)
            }
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        if DEBUG_MODE:
            print(f"Generated price data for {symbol}: {len(df)} rows")
            if not df.empty:
                print(f"Price data columns: {df.columns.tolist()}")
                print(f"Price data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df

    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1):
        """
        Get historical data for a symbol using Schwab API
        
        Args:
            symbol (str): The stock symbol to get historical data for
            period_type (str): The type of period to show (day, month, year, ytd)
            period (int): The number of periods to show
            frequency_type (str): The type of frequency (minute, daily, weekly, monthly)
            frequency (int): The frequency value
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            if DEBUG_MODE:
                print(f"Getting historical data for {symbol} using Schwab API")
                print(f"Parameters: period_type={period_type}, period={period}, frequency_type={frequency_type}, frequency={frequency}")
            
            # Get the underlying symbol if this is an option
            underlying_symbol = self.get_underlying_symbol(symbol)
            
            # Call Schwab API's price_history method
            history_response = self.client.price_history(
                symbol=underlying_symbol,
                period_type=period_type,
                period=period,
                frequency_type=frequency_type,
                frequency=frequency
            )
            
            if DEBUG_MODE:
                print(f"Price history response type: {type(history_response)}")
                if hasattr(history_response, 'status_code'):
                    print(f"Status code: {history_response.status_code}")
            
            # Process the response
            history_data = None
            if hasattr(history_response, 'json'):
                try:
                    history_data = history_response.json()
                    if DEBUG_MODE:
                        print(f"Price history received for {underlying_symbol}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error parsing price history JSON: {str(e)}")
                        if hasattr(history_response, 'text'):
                            print(f"Response text: {history_response.text[:500]}...")
            elif isinstance(history_response, dict):
                history_data = history_response
                if DEBUG_MODE:
                    print(f"Price history received for {underlying_symbol}")
            else:
                if DEBUG_MODE:
                    print(f"No price history data received for {underlying_symbol}")
                    
            # Convert to DataFrame
            if history_data and 'candles' in history_data:
                candles = history_data['candles']
                df = pd.DataFrame(candles)
                
                # Rename columns to match expected format
                column_mapping = {
                    'datetime': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }
                
                # Apply column mapping if needed
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                # Convert timestamp to datetime if it's in milliseconds
                if 'timestamp' in df.columns and df['timestamp'].dtype == 'int64':
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                if DEBUG_MODE:
                    print(f"Processed historical data: {len(df)} rows")
                    if not df.empty:
                        print(f"Historical data columns: {df.columns.tolist()}")
                        print(f"Historical data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                return df
            else:
                if DEBUG_MODE:
                    print(f"No candles found in price history data for {underlying_symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                print("Falling back to get_price_data")
            
            # Fall back to get_price_data if there's an error
            return self.get_price_data(symbol, period_type, period, frequency_type, frequency)
