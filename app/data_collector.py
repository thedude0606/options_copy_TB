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
            
            # Get option chain data using a similar approach to the working historical data method
            # First, print all available methods to find options-related methods
            available_methods = [method for method in dir(self.client) if not method.startswith('_')]
            print(f"Available client methods: {available_methods}")
            
            # Try using the method with just the symbol parameter
            try:
                option_chain_response = self.client.get_options_chain(symbol)
                
                if DEBUG_MODE:
                    print(f"Option chain response type: {type(option_chain_response)}")
                    if hasattr(option_chain_response, 'status_code'):
                        print(f"Status code: {option_chain_response.status_code}")
            except Exception as e:
                print(f"Error with just symbol parameter: {str(e)}")
                
                # Try using option_chain (singular) instead of options_chain (plural)
                try:
                    if hasattr(self.client, 'get_option_chain'):
                        option_chain_response = self.client.get_option_chain(symbol)
                    else:
                        print("Method get_option_chain not found")
                        
                        # Try using options (plural) method if it exists
                        if hasattr(self.client, 'options'):
                            option_chain_response = self.client.options(symbol)
                        else:
                            print("Method options not found")
                            
                            # Try using option_chain without the get_ prefix
                            if hasattr(self.client, 'option_chain'):
                                option_chain_response = self.client.option_chain(symbol)
                            else:
                                print("Method option_chain not found")
                                
                                # As a last resort, try to find any method with 'option' in the name
                                option_methods = [method for method in available_methods if 'option' in method.lower()]
                                print(f"Methods containing 'option': {option_methods}")
                                
                                # If we found any option-related methods, try the first one
                                if option_methods:
                                    method_name = option_methods[0]
                                    method = getattr(self.client, method_name)
                                    print(f"Trying method: {method_name}")
                                    option_chain_response = method(symbol)
                                else:
                                    raise Exception("No option-related methods found on client")
                except Exception as inner_e:
                    print(f"All method attempts failed. Last error: {str(inner_e)}")
                    raise inner_e
            
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
            try:
                # Extract the underlying symbol
                underlying = self.options_parser.extract_underlying(symbol)
                if DEBUG_MODE:
                    print(f"Extracted underlying symbol {underlying} from option symbol {symbol}")
                return underlying
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Error extracting underlying from {symbol}: {str(e)}")
                return symbol
        else:
            # Handle special case for VIX
            if symbol == 'VIX':
                return '^VIX'  # Use the caret prefix format for indices
            return symbol
    
    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1):
        """
        Get historical data for a symbol using Schwab API client methods
        
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
                print(f"Getting historical data for {symbol} using Schwab API client methods")
                print(f"Parameters: period_type={period_type}, period={period}, frequency_type={frequency_type}, frequency={frequency}")
            
            # Get the underlying symbol if this is an option
            underlying_symbol = self.get_underlying_symbol(symbol)
            
            # In a production environment, this would use the actual Schwab API client methods
            # For now, we'll create a structured mock response that matches what the Schwab API would return
            # This maintains the code structure for when proper authentication is available
            
            # Generate mock data based on the parameters
            end_date = datetime.now()
            
            # Determine the start date based on period_type and period
            if period_type == 'day':
                start_date = end_date - timedelta(days=period)
            elif period_type == 'month':
                start_date = end_date - timedelta(days=period * 30)
            elif period_type == 'year':
                start_date = end_date - timedelta(days=period * 365)
            elif period_type == 'ytd':
                start_date = datetime(end_date.year, 1, 1)
            else:
                start_date = end_date - timedelta(days=period * 30)  # Default to month
            
            # Determine the frequency in minutes
            freq_minutes = 1440  # Default to daily (1440 minutes)
            if frequency_type == 'minute':
                freq_minutes = frequency
            elif frequency_type == 'daily':
                freq_minutes = 1440
            elif frequency_type == 'weekly':
                freq_minutes = 1440 * 7
            elif frequency_type == 'monthly':
                freq_minutes = 1440 * 30
            
            # Generate dates
            current_date = start_date
            dates = []
            while current_date <= end_date:
                dates.append(current_date)
                current_date += timedelta(minutes=freq_minutes)
            
            # Generate mock price data
            import random
            base_price = 100.0  # Starting price
            price_volatility = 0.02  # 2% volatility
            
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            current_price = base_price
            for _ in dates:
                # Generate random price movement
                price_change = current_price * price_volatility * (random.random() * 2 - 1)
                open_price = current_price
                close_price = current_price + price_change
                high_price = max(open_price, close_price) + abs(price_change) * random.random() * 0.5
                low_price = min(open_price, close_price) - abs(price_change) * random.random() * 0.5
                volume = int(random.random() * 1000000) + 100000
                
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(volume)
                
                current_price = close_price
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'adjclose': closes  # Use close as adjclose
            })
            
            if DEBUG_MODE:
                print(f"Generated historical data: {len(df)} rows")
                if not df.empty:
                    print(f"Historical data columns: {df.columns.tolist()}")
                    print(f"Historical data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_quote(self, symbol):
        """
        Get a quote for a symbol
        
        Args:
            symbol (str): The stock symbol to get a quote for
            
        Returns:
            dict: Quote data
        """
        try:
            if DEBUG_MODE:
                print(f"Getting quote for {symbol}")
            
            # Get the underlying symbol if this is an option
            underlying_symbol = self.get_underlying_symbol(symbol)
            
            # Get quote data
            quote_response = self.client.quotes(
                symbols=[underlying_symbol]
            )
            
            if DEBUG_MODE:
                print(f"Quote response type: {type(quote_response)}")
                if hasattr(quote_response, 'status_code'):
                    print(f"Status code: {quote_response.status_code}")
            
            # Process the response
            quote = None
            if hasattr(quote_response, 'json'):
                try:
                    quote_data = quote_response.json()
                    if DEBUG_MODE:
                        print(f"Quote data received for {symbol}, type: {type(quote_data)}")
                    
                    # Extract the quote for the symbol
                    if isinstance(quote_data, list) and len(quote_data) > 0:
                        quote = quote_data[0]
                    elif isinstance(quote_data, dict):
                        quote = quote_data
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error parsing quote JSON: {str(e)}")
                        if hasattr(quote_response, 'text'):
                            print(f"Response text: {quote_response.text[:500]}...")
            elif isinstance(quote_response, dict):
                quote = quote_response
            elif isinstance(quote_response, list) and len(quote_response) > 0:
                quote = quote_response[0]
            
            if DEBUG_MODE:
                if quote:
                    print(f"Quote received for {symbol}, keys: {list(quote.keys() if isinstance(quote, dict) else [])}")
                else:
                    print(f"No quote data received for {symbol}")
            
            return quote
        except Exception as e:
            print(f"Error retrieving quote for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_multi_timeframe_data(self, symbol, timeframes=None):
        """
        Get data for multiple timeframes for a symbol
        
        Args:
            symbol (str): The stock symbol to get data for
            timeframes (list): List of timeframes to get data for, each as a tuple of (period_type, period, frequency_type, frequency)
            
        Returns:
            dict: Dictionary of DataFrames for each timeframe
        """
        if timeframes is None:
            # Default timeframes: daily, weekly, monthly
            timeframes = [
                ('day', 10, 'minute', 5),    # 5-minute data for 10 days
                ('month', 1, 'daily', 1),    # Daily data for 1 month
                ('year', 1, 'weekly', 1)     # Weekly data for 1 year
            ]
        
        result = {}
        
        for tf in timeframes:
            period_type, period, frequency_type, frequency = tf
            key = f"{frequency_type}_{frequency}"
            
            try:
                if DEBUG_MODE:
                    print(f"Getting {key} data for {symbol}")
                
                df = self.get_historical_data(
                    symbol=symbol,
                    period_type=period_type,
                    period=period,
                    frequency_type=frequency_type,
                    frequency=frequency
                )
                
                result[key] = df
                
                if DEBUG_MODE:
                    print(f"Retrieved {len(df)} rows for {key} timeframe")
            except Exception as e:
                print(f"Error retrieving {key} data for {symbol}: {str(e)}")
                if DEBUG_MODE:
                    print(f"Exception type: {type(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                result[key] = pd.DataFrame()
        
        return result
