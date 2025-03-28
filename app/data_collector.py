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
    
    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', 
                           frequency=1, need_extended_hours_data=True):
        """
        Get historical price data for a symbol with retry logic
        
        Args:
            symbol (str): The stock symbol or option symbol
            period_type (str): Type of period - 'day', 'month', 'year', 'ytd'
            period (int): Number of periods
            frequency_type (str): Type of frequency - 'minute', 'daily', 'weekly', 'monthly'
            frequency (int): Frequency
            need_extended_hours_data (bool): Whether to include extended hours data
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            if DEBUG_MODE:
                print(f"\n=== HISTORICAL DATA REQUEST ===")
                print(f"Symbol: {symbol}")
                print(f"Parameters: periodType={period_type}, period={period}, frequencyType={frequency_type}, frequency={frequency}")
            
            # Extract underlying symbol if this is an option
            original_symbol = symbol
            symbol = self.get_underlying_symbol(symbol)
            
            if symbol != original_symbol and DEBUG_MODE:
                print(f"Using underlying symbol {symbol} instead of option symbol {original_symbol}")
            
            # Check cache first
            cache_key = f"{symbol}_{period_type}_{period}_{frequency_type}_{frequency}"
            if cache_key in self.cache:
                if DEBUG_MODE:
                    print(f"Using cached data for {symbol}")
                return self.cache[cache_key]
            
            # Validate and correct parameters before API call
            # Ensure period_type is valid
            valid_period_types = ['day', 'month', 'year', 'ytd']
            if period_type not in valid_period_types:
                if DEBUG_MODE:
                    print(f"Warning: Invalid period_type '{period_type}'. Defaulting to 'day'")
                period_type = 'day'
                
            # Ensure frequency_type is valid and compatible with period_type
            valid_frequency_types = {
                'day': ['minute'],
                'month': ['daily', 'weekly'],
                'year': ['daily', 'weekly', 'monthly'],
                'ytd': ['daily', 'weekly']
            }
            
            if frequency_type not in valid_frequency_types.get(period_type, []):
                if DEBUG_MODE:
                    print(f"Warning: Incompatible frequency_type '{frequency_type}' for period_type '{period_type}'")
                # Set compatible defaults
                if period_type == 'day':
                    frequency_type = 'minute'
                else:
                    frequency_type = 'daily'
                    
            # Ensure frequency is valid for the frequency_type
            valid_frequencies = {
                'minute': [1, 5, 10, 15, 30],
                'daily': [1],
                'weekly': [1],
                'monthly': [1]
            }
            
            if frequency not in valid_frequencies.get(frequency_type, []):
                if DEBUG_MODE:
                    print(f"Warning: Invalid frequency '{frequency}' for frequency_type '{frequency_type}'")
                # Set to default valid frequency
                frequency = valid_frequencies.get(frequency_type, [1])[0]
                
            if VERBOSE_DEBUG:
                print(f"Validated parameters: periodType={period_type}, period={period}, frequencyType={frequency_type}, frequency={frequency}")
            
            # Get historical price data with retry logic
            history = None
            
            # Try with primary parameters - using camelCase parameter names
            try:
                if DEBUG_MODE:
                    print(f"Attempting primary request with camelCase parameters...")
                
                history = self.client.price_history(
                    symbol=symbol,
                    periodType=period_type,
                    period=period,
                    frequencyType=frequency_type,
                    frequency=frequency,
                    needExtendedHoursData=need_extended_hours_data
                )
                
                if DEBUG_MODE:
                    print(f"Primary request response type: {type(history)}")
                    if hasattr(history, 'status_code'):
                        print(f"Status code: {history.status_code}")
            except Exception as e:
                print(f"Primary parameters failed: {str(e)}")
                if DEBUG_MODE:
                    print(f"Exception type: {type(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
            
            # If primary parameters failed, try alternative configurations
            if not history:
                try:
                    if DEBUG_MODE:
                        print(f"Attempting alternative request with daily frequency...")
                    
                    # Try with daily frequency - using camelCase parameter names
                    history = self.client.price_history(
                        symbol=symbol,
                        periodType='month',
                        period=1,
                        frequencyType='daily',
                        frequency=1,
                        needExtendedHoursData=need_extended_hours_data
                    )
                    
                    if DEBUG_MODE:
                        print(f"Alternative request response type: {type(history)}")
                        if hasattr(history, 'status_code'):
                            print(f"Status code: {history.status_code}")
                except Exception as e:
                    print(f"Alternative parameters failed: {str(e)}")
                    if DEBUG_MODE:
                        print(f"Exception type: {type(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
            
            # Try a third approach with minimal parameters
            if not history:
                try:
                    if DEBUG_MODE:
                        print(f"Attempting minimal parameter request...")
                    
                    # Try with minimal parameters
                    history = self.client.price_history(symbol=symbol)
                    
                    if DEBUG_MODE:
                        print(f"Minimal request response type: {type(history)}")
                        if hasattr(history, 'status_code'):
                            print(f"Status code: {history.status_code}")
                except Exception as e:
                    print(f"Minimal parameters failed: {str(e)}")
                    if DEBUG_MODE:
                        print(f"Exception type: {type(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
            
            # Process historical data
            if history:
                if DEBUG_MODE:
                    print(f"Processing history response...")
                    print(f"Response has json method: {hasattr(history, 'json')}")
                    print(f"Response has text attribute: {hasattr(history, 'text')}")
                
                history_data = None
                
                # Try to get JSON data
                if hasattr(history, 'json'):
                    try:
                        history_data = history.json()
                        if DEBUG_MODE:
                            print(f"JSON data keys: {list(history_data.keys() if isinstance(history_data, dict) else [])}")
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"Error parsing JSON: {str(e)}")
                            if hasattr(history, 'text'):
                                print(f"Response text: {history.text[:500]}...")
                
                # If we couldn't get JSON data but have a dict, use it directly
                if not history_data and isinstance(history, dict):
                    history_data = history
                    if DEBUG_MODE:
                        print(f"Using dict response directly, keys: {list(history_data.keys())}")
                
                # Extract candles
                candles = []
                if history_data:
                    candles = history_data.get('candles', [])
                    if DEBUG_MODE:
                        print(f"Found {len(candles)} candles")
                
                if candles:
                    # Convert to DataFrame
                    df = pd.DataFrame(candles)
                    
                    if DEBUG_MODE:
                        print(f"DataFrame columns: {list(df.columns)}")
                        print(f"DataFrame shape: {df.shape}")
                    
                    # Convert datetime
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                    
                    # Set datetime as index
                    if 'datetime' in df.columns:
                        df.set_index('datetime', inplace=True)
                    
                    # Add symbol column
                    df['symbol'] = symbol
                    
                    # Add original_symbol column if it was an option
                    if symbol != original_symbol:
                        df['original_symbol'] = original_symbol
                    
                    # Cache the result
                    self.cache[cache_key] = df
                    
                    return df
                else:
                    print(f"No candles data in response for {symbol}")
                    if DEBUG_MODE and history_data:
                        print(f"Response data: {json.dumps(history_data, indent=2)[:500]}...")
                    return pd.DataFrame()
            else:
                print(f"No valid historical data returned for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_multi_timeframe_data(self, symbol, timeframes=None):
        """
        Get historical data for multiple timeframes
        
        Args:
            symbol (str): The stock symbol or option symbol
            timeframes (list): List of timeframe configurations, each as a dict with
                              period_type, period, frequency_type, and frequency
                              
        Returns:
            dict: Dictionary of DataFrames, keyed by timeframe name
        """
        if timeframes is None:
            # Default timeframes
            timeframes = [
                {
                    'name': 'daily',
                    'period_type': 'month',
                    'period': 3,
                    'frequency_type': 'daily',
                    'frequency': 1
                },
                {
                    'name': 'weekly',
                    'period_type': 'year',
                    'period': 1,
                    'frequency_type': 'weekly',
                    'frequency': 1
                },
                {
                    'name': 'monthly',
                    'period_type': 'year',
                    'period': 3,
                    'frequency_type': 'monthly',
                    'frequency': 1
                }
            ]
        
        result = {}
        
        for tf in timeframes:
            name = tf.get('name', f"{tf['period_type']}_{tf['period']}_{tf['frequency_type']}_{tf['frequency']}")
            
            if DEBUG_MODE:
                print(f"Getting {name} data for {symbol}")
                
            data = self.get_historical_data(
                symbol=symbol,
                period_type=tf['period_type'],
                period=tf['period'],
                frequency_type=tf['frequency_type'],
                frequency=tf['frequency']
            )
            
            result[name] = data
            
        return result
    
    def get_quote(self, symbol):
        """
        Get current quote for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Quote data
        """
        try:
            # Check if this is an option symbol and get the underlying if needed
            if self.options_parser.is_option_symbol(symbol):
                original_symbol = symbol
                symbol = self.options_parser.get_underlying_symbol(symbol)
                if DEBUG_MODE:
                    print(f"Requesting quote for symbol: {symbol} (extracted from {original_symbol})")
            else:
                if DEBUG_MODE:
                    print(f"Requesting quote for symbol: {symbol}")
            
            # Get quote data
            quote = self.client.quote(symbol)
            
            if quote:
                if DEBUG_MODE:
                    print(f"Quote received for {symbol}")
                    if isinstance(quote, dict):
                        print(f"Quote data keys: {list(quote.keys())}")
            else:
                if DEBUG_MODE:
                    print(f"No quote data received for {symbol}")
                
                # Try with caret prefix for indices like VIX
                if symbol == "VIX":
                    if DEBUG_MODE:
                        print(f"Trying with caret prefix: ^{symbol}")
                    quote = self.client.quote(f"^{symbol}")
                    if quote and DEBUG_MODE:
                        print(f"Quote received for ^{symbol}")
                        if isinstance(quote, dict):
                            print(f"Quote data keys: {list(quote.keys())}")
            
            return quote
        except Exception as e:
            print(f"Error retrieving quote for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
