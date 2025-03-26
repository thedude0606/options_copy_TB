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
    
    def get_options_chain(self, symbol):
        """
        Get the options chain for a symbol (plural version for compatibility)
        
        Args:
            symbol (str): The stock symbol to get options for
            
        Returns:
            dict: Option chain data
        """
        # This is a wrapper around get_option_chain to match the expected method name
        # in the error logs and provide compatibility with both naming conventions
        return self.get_option_chain_with_underlying_price(symbol)
    
    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', 
                           frequency=1, need_extended_hours_data=True):
        """
        Get historical price data for a symbol with retry logic
        
        Args:
            symbol (str): The stock symbol
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
                        periodType='day',
                        period=1,
                        frequencyType='minute',
                        frequency=30,
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
            
            # If still no history, try one more time with minimal parameters
            if not history:
                try:
                    if DEBUG_MODE:
                        print(f"Attempting minimal request with symbol only...")
                    
                    # Try with just the symbol
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
            
            # Process the response
            if history:
                if hasattr(history, 'json'):
                    try:
                        history_data = history.json()
                        if DEBUG_MODE:
                            print(f"History data keys: {list(history_data.keys() if isinstance(history_data, dict) else [])}")
                        
                        # Extract candles from the response
                        candles = history_data.get('candles', [])
                        if candles:
                            # Convert to DataFrame
                            df = pd.DataFrame(candles)
                            
                            # Rename columns to match expected format
                            column_mapping = {
                                'datetime': 'datetime',
                                'open': 'open',
                                'high': 'high',
                                'low': 'low',
                                'close': 'close',
                                'volume': 'volume'
                            }
                            
                            # Ensure all required columns exist
                            for old_col, new_col in column_mapping.items():
                                if old_col not in df.columns:
                                    df[new_col] = np.nan
                                elif old_col != new_col:
                                    df[new_col] = df[old_col]
                                    df.drop(columns=[old_col], inplace=True)
                            
                            # Convert datetime from milliseconds to datetime objects
                            if 'datetime' in df.columns:
                                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                            
                            return df
                        else:
                            if DEBUG_MODE:
                                print(f"No candles found in history data")
                            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                    except Exception as e:
                        print(f"Error processing history data: {str(e)}")
                        if DEBUG_MODE:
                            print(f"Exception type: {type(e)}")
                            print(f"Traceback: {traceback.format_exc()}")
                        return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                elif isinstance(history, dict):
                    # Extract candles from the response
                    candles = history.get('candles', [])
                    if candles:
                        # Convert to DataFrame
                        df = pd.DataFrame(candles)
                        
                        # Rename columns to match expected format
                        column_mapping = {
                            'datetime': 'datetime',
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'volume': 'volume'
                        }
                        
                        # Ensure all required columns exist
                        for old_col, new_col in column_mapping.items():
                            if old_col not in df.columns:
                                df[new_col] = np.nan
                            elif old_col != new_col:
                                df[new_col] = df[old_col]
                                df.drop(columns=[old_col], inplace=True)
                        
                        # Convert datetime from milliseconds to datetime objects
                        if 'datetime' in df.columns:
                            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                        
                        return df
                    else:
                        if DEBUG_MODE:
                            print(f"No candles found in history data")
                        return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                else:
                    if DEBUG_MODE:
                        print(f"Unexpected history data type: {type(history)}")
                    return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            else:
                if DEBUG_MODE:
                    print(f"No history data returned")
                return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_quote(self, symbol):
        """
        Get current quote for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Quote data
        """
        try:
            if DEBUG_MODE:
                print(f"Requesting quote for symbol: {symbol}")
            
            # Get quote data
            quote_response = self.client.quote(symbol)
            
            if DEBUG_MODE:
                if quote_response:
                    print(f"Quote received for {symbol}")
                else:
                    print(f"No quote data received for {symbol}")
            
            # Process the response
            quote_data = None
            
            # Check if response has json method (Response object)
            if hasattr(quote_response, 'json'):
                # Check if response is successful
                if hasattr(quote_response, 'ok') and quote_response.ok:
                    try:
                        quote_data = quote_response.json()
                        if DEBUG_MODE:
                            print(f"Quote data keys: {list(quote_data.keys() if isinstance(quote_data, dict) else [])}")
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"Error parsing quote JSON: {str(e)}")
                            if hasattr(quote_response, 'text'):
                                print(f"Response text: {quote_response.text[:500]}...")
                else:
                    if DEBUG_MODE:
                        status_code = getattr(quote_response, 'status_code', 'unknown')
                        print(f"Quote response not OK. Status code: {status_code}")
            # If response is already a dict, use it directly
            elif isinstance(quote_response, dict):
                quote_data = quote_response
                if DEBUG_MODE:
                    print(f"Quote data keys: {list(quote_data.keys())}")
            
            return quote_data
        except Exception as e:
            print(f"Error retrieving quote for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_market_hours(self, market='EQUITY'):
        """
        Get market hours
        
        Args:
            market (str): Market to get hours for (EQUITY, OPTION, BOND, FOREX)
            
        Returns:
            dict: Market hours data
        """
        try:
            if DEBUG_MODE:
                print(f"Requesting market hours for: {market}")
            
            # Get market hours data
            hours_response = self.client.get_market_hours(market=market)
            
            # Process the response
            hours_data = None
            if hasattr(hours_response, 'json'):
                try:
                    hours_data = hours_response.json()
                    if DEBUG_MODE:
                        print(f"Market hours received for {market}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error parsing market hours JSON: {str(e)}")
            elif isinstance(hours_response, dict):
                hours_data = hours_response
                if DEBUG_MODE:
                    print(f"Market hours received for {market}")
            else:
                if DEBUG_MODE:
                    print(f"No market hours data received for {market}")
            
            return hours_data
        except Exception as e:
            print(f"Error retrieving market hours for {market}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_price_history(self, symbol, period_type=None, period=None, frequency_type=None, 
                         frequency=None, start_date=None, end_date=None, need_extended_hours_data=True):
        """
        Get price history for a symbol
        
        Args:
            symbol (str): The stock symbol
            period_type (str, optional): Type of period - 'day', 'month', 'year', 'ytd'
            period (int, optional): Number of periods
            frequency_type (str, optional): Type of frequency - 'minute', 'daily', 'weekly', 'monthly'
            frequency (int, optional): Frequency
            start_date (datetime or str, optional): Start date for custom date range
            end_date (datetime or str, optional): End date for custom date range
            need_extended_hours_data (bool): Whether to include extended hours data
            
        Returns:
            pd.DataFrame: Price history data
        """
        # This is a wrapper around get_historical_data to match the expected method name
        # in the error logs and provide compatibility with both naming conventions
        return self.get_historical_data(
            symbol=symbol,
            period_type=period_type if period_type else 'day',
            period=period if period else 10,
            frequency_type=frequency_type if frequency_type else 'minute',
            frequency=frequency if frequency else 1,
            need_extended_hours_data=need_extended_hours_data
        )
    
    def get_option_chain_with_underlying_price(self, symbol):
        """
        Get option chain with underlying price
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Option chain data with underlying price
        """
        try:
            if DEBUG_MODE:
                print(f"Requesting option chain with underlying price for: {symbol}")
            
            # Get option chain
            option_chain = self.get_option_chain(symbol)
            
            # If no option chain or no underlying price, try to get it from quote
            if not option_chain or 'underlying_price' not in option_chain or not option_chain['underlying_price']:
                if DEBUG_MODE:
                    print(f"No underlying price in option chain, trying to get quote")
                
                # Get quote to get current price
                quote = self.get_quote(symbol)
                if quote and isinstance(quote, dict):
                    if 'mark' in quote:
                        underlying_price = quote['mark']
                        if DEBUG_MODE:
                            print(f"Using underlying price from quote 'mark': {underlying_price}")
                    elif 'lastPrice' in quote:
                        underlying_price = quote['lastPrice']
                        if DEBUG_MODE:
                            print(f"Using underlying price from quote 'lastPrice': {underlying_price}")
                    else:
                        underlying_price = None
                        if DEBUG_MODE:
                            print(f"No suitable price found in quote")
                    
                    # Add underlying price to option chain
                    if underlying_price and option_chain:
                        option_chain['underlying_price'] = underlying_price
            
            return option_chain
        except Exception as e:
            print(f"Error retrieving option chain with underlying price for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
