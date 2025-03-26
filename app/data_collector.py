"""
Data collection module for options recommendation platform.
Handles retrieving market data from Schwab API for technical indicators and options analysis.
"""
import os
import pandas as pd
import numpy as np
import traceback
import json
import requests
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
    
    def get_market_data(self, symbol=None):
        """
        Get market data for dashboard or for a specific symbol
        
        Args:
            symbol (str, optional): Symbol to get data for. If None, returns data for major indices.
            
        Returns:
            dict: Market data
        """
        try:
            # If a specific symbol is requested, get data just for that symbol
            if symbol:
                try:
                    print(f"Requesting quote for symbol: {symbol}")
                    quote_response = self.client.quote(symbol)
                    
                    # Enhanced debugging for quote response
                    if VERBOSE_DEBUG:
                        print(f"Quote response type: {type(quote_response)}")
                    
                    # Check if response is a requests.Response object
                    if isinstance(quote_response, requests.Response):
                        if quote_response.status_code == 200:
                            try:
                                quote_data = quote_response.json()
                                print(f"Quote received for {symbol}")
                                if VERBOSE_DEBUG:
                                    print(f"Quote data keys: {list(quote_data.keys() if isinstance(quote_data, dict) else [])}")
                                
                                # The Schwab API returns data in a specific format
                                # The actual quote data might be nested under the symbol key
                                if symbol in quote_data:
                                    symbol_data = quote_data[symbol]
                                    return {
                                        'lastPrice': symbol_data.get('lastPrice', 0),
                                        'netChange': symbol_data.get('netChange', 0),
                                        'netPercentChangeInDouble': symbol_data.get('netPercentChangeInDouble', 0),
                                        'totalVolume': symbol_data.get('totalVolume', 0),
                                        'description': symbol_data.get('description', symbol)
                                    }
                                else:
                                    # If not nested, try to get data directly
                                    return {
                                        'lastPrice': quote_data.get('lastPrice', 0),
                                        'netChange': quote_data.get('netChange', 0),
                                        'netPercentChangeInDouble': quote_data.get('netPercentChangeInDouble', 0),
                                        'totalVolume': quote_data.get('totalVolume', 0),
                                        'description': quote_data.get('description', symbol)
                                    }
                            except ValueError as e:
                                print(f"Error parsing JSON for {symbol}: {str(e)}")
                                # Try to get the raw text for debugging
                                if hasattr(quote_response, 'text'):
                                    print(f"Response text: {quote_response.text[:200]}...")
                        else:
                            print(f"Quote response not OK for {symbol}. Status code: {quote_response.status_code}")
                    elif isinstance(quote_response, dict):
                        # If the client already returned a dict instead of a Response object
                        print(f"Quote response is already a dict for {symbol}")
                        if symbol in quote_response:
                            symbol_data = quote_response[symbol]
                            return {
                                'lastPrice': symbol_data.get('lastPrice', 0),
                                'netChange': symbol_data.get('netChange', 0),
                                'netPercentChangeInDouble': symbol_data.get('netPercentChangeInDouble', 0),
                                'totalVolume': symbol_data.get('totalVolume', 0),
                                'description': symbol_data.get('description', symbol)
                            }
                        else:
                            return {
                                'lastPrice': quote_response.get('lastPrice', 0),
                                'netChange': quote_response.get('netChange', 0),
                                'netPercentChangeInDouble': quote_response.get('netPercentChangeInDouble', 0),
                                'totalVolume': quote_response.get('totalVolume', 0),
                                'description': quote_response.get('description', symbol)
                            }
                    else:
                        print(f"Unexpected quote response type for {symbol}: {type(quote_response)}")
                        print(f"Quote response: {quote_response}")
                except Exception as e:
                    print(f"Error getting quote for {symbol}: {str(e)}")
                    if VERBOSE_DEBUG:
                        print(traceback.format_exc())
                    
                    # Return empty data if we couldn't get a quote
                    return {
                        'lastPrice': 0,
                        'netChange': 0,
                        'netPercentChangeInDouble': 0,
                        'totalVolume': 0,
                        'description': symbol
                    }
            
            # If no specific symbol is requested, get data for major indices
            else:
                # Get quotes for major indices
                indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']
                quotes = {}
                
                for idx_symbol in indices:
                    try:
                        print(f"Requesting quote for symbol: {idx_symbol}")
                        quote_response = self.client.quote(idx_symbol)
                        
                        # Enhanced debugging for quote response
                        if VERBOSE_DEBUG:
                            print(f"Quote response type: {type(quote_response)}")
                        
                        # Check if response is a requests.Response object
                        if isinstance(quote_response, requests.Response):
                            if quote_response.status_code == 200:
                                try:
                                    quote_data = quote_response.json()
                                    print(f"Quote received for {idx_symbol}")
                                    if VERBOSE_DEBUG:
                                        print(f"Quote data keys: {list(quote_data.keys() if isinstance(quote_data, dict) else [])}")
                                    
                                    # The Schwab API returns data in a specific format
                                    # The actual quote data might be nested under the symbol key
                                    if idx_symbol in quote_data:
                                        symbol_data = quote_data[idx_symbol]
                                        quotes[idx_symbol] = {
                                            'lastPrice': symbol_data.get('lastPrice', 0),
                                            'netChange': symbol_data.get('netChange', 0),
                                            'netPercentChangeInDouble': symbol_data.get('netPercentChangeInDouble', 0),
                                            'totalVolume': symbol_data.get('totalVolume', 0),
                                            'description': symbol_data.get('description', idx_symbol)
                                        }
                                    else:
                                        # If not nested, try to get data directly
                                        quotes[idx_symbol] = {
                                            'lastPrice': quote_data.get('lastPrice', 0),
                                            'netChange': quote_data.get('netChange', 0),
                                            'netPercentChangeInDouble': quote_data.get('netPercentChangeInDouble', 0),
                                            'totalVolume': quote_data.get('totalVolume', 0),
                                            'description': quote_data.get('description', idx_symbol)
                                        }
                                except ValueError as e:
                                    print(f"Error parsing JSON for {idx_symbol}: {str(e)}")
                                    # Try to get the raw text for debugging
                                    if hasattr(quote_response, 'text'):
                                        print(f"Response text: {quote_response.text[:200]}...")
                            else:
                                print(f"Quote response not OK. Status code: {quote_response.status_code}")
                        elif isinstance(quote_response, dict):
                            # If the client already returned a dict instead of a Response object
                            print(f"Quote received for {idx_symbol}")
                            if VERBOSE_DEBUG:
                                print(f"Quote data keys: {list(quote_response.keys() if isinstance(quote_response, dict) else [])}")
                            
                            if idx_symbol in quote_response:
                                symbol_data = quote_response[idx_symbol]
                                quotes[idx_symbol] = {
                                    'lastPrice': symbol_data.get('lastPrice', 0),
                                    'netChange': symbol_data.get('netChange', 0),
                                    'netPercentChangeInDouble': symbol_data.get('netPercentChangeInDouble', 0),
                                    'totalVolume': symbol_data.get('totalVolume', 0),
                                    'description': symbol_data.get('description', idx_symbol)
                                }
                            else:
                                quotes[idx_symbol] = {
                                    'lastPrice': quote_response.get('lastPrice', 0),
                                    'netChange': quote_response.get('netChange', 0),
                                    'netPercentChangeInDouble': quote_response.get('netPercentChangeInDouble', 0),
                                    'totalVolume': quote_response.get('totalVolume', 0),
                                    'description': quote_response.get('description', idx_symbol)
                                }
                        else:
                            print(f"No quote data received for {idx_symbol}")
                    except Exception as e:
                        print(f"Error getting quote for {idx_symbol}: {str(e)}")
                        if VERBOSE_DEBUG:
                            print(traceback.format_exc())
                
                # Get market calendar
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Get market movers (most active, gainers, losers)
                # This would typically come from a market movers API
                # For now, we'll just use placeholder data
                market_movers = {
                    'most_active': [
                        {'symbol': 'AAPL', 'lastPrice': 150.0, 'netPercentChangeInDouble': 1.5, 'totalVolume': 80000000},
                        {'symbol': 'MSFT', 'lastPrice': 290.0, 'netPercentChangeInDouble': 0.8, 'totalVolume': 30000000},
                        {'symbol': 'TSLA', 'lastPrice': 200.0, 'netPercentChangeInDouble': -2.1, 'totalVolume': 70000000}
                    ],
                    'gainers': [
                        {'symbol': 'XYZ', 'lastPrice': 45.0, 'netPercentChangeInDouble': 15.0, 'totalVolume': 5000000},
                        {'symbol': 'ABC', 'lastPrice': 30.0, 'netPercentChangeInDouble': 12.5, 'totalVolume': 3000000},
                        {'symbol': 'DEF', 'lastPrice': 75.0, 'netPercentChangeInDouble': 10.2, 'totalVolume': 2000000}
                    ],
                    'losers': [
                        {'symbol': 'UVW', 'lastPrice': 80.0, 'netPercentChangeInDouble': -18.0, 'totalVolume': 4000000},
                        {'symbol': 'RST', 'lastPrice': 15.0, 'netPercentChangeInDouble': -15.3, 'totalVolume': 2500000},
                        {'symbol': 'MNO', 'lastPrice': 45.0, 'netPercentChangeInDouble': -12.1, 'totalVolume': 1800000}
                    ]
                }
                
                return {
                    'quotes': quotes,
                    'date': today,
                    'market_movers': market_movers
                }
        except Exception as e:
            print(f"Error retrieving market data: {str(e)}")
            if VERBOSE_DEBUG:
                print(traceback.format_exc())
            
            # Return empty data structure
            if symbol:
                return {
                    'lastPrice': 0,
                    'netChange': 0,
                    'netPercentChangeInDouble': 0,
                    'totalVolume': 0,
                    'description': symbol
                }
            else:
                return {
                    'quotes': {},
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'market_movers': {
                        'most_active': [],
                        'gainers': [],
                        'losers': []
                    }
                }
    
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
            if isinstance(option_chain_response, requests.Response):
                if option_chain_response.status_code == 200:
                    try:
                        option_chain = option_chain_response.json()
                        if DEBUG_MODE:
                            print(f"Option chain received for {symbol}, keys: {list(option_chain.keys() if isinstance(option_chain, dict) else [])}")
                            if isinstance(option_chain, dict) and 'underlying' in option_chain:
                                print(f"Option chain structure for {symbol}:")
                                print(f"Top-level keys: {list(option_chain.keys())}")
                        return option_chain
                    except ValueError as e:
                        print(f"Error parsing JSON for option chain: {str(e)}")
                        if DEBUG_MODE and hasattr(option_chain_response, 'text'):
                            print(f"Response text: {option_chain_response.text[:200]}...")
                else:
                    print(f"Option chain response not OK. Status code: {option_chain_response.status_code}")
            elif isinstance(option_chain_response, dict):
                # If the client already returned a dict instead of a Response object
                if DEBUG_MODE:
                    print(f"Option chain response is already a dict for {symbol}")
                return option_chain_response
            else:
                print(f"Unexpected option chain response type: {type(option_chain_response)}")
            
            return None
        except Exception as e:
            print(f"Error retrieving option chain for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
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
            dict: Option chain data with underlying price
        """
        # This is a wrapper around get_option_chain_with_underlying_price to match the expected method name
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
            
            # Process the response
            if history:
                if hasattr(history, 'json'):
                    try:
                        history_data = history.json()
                        if DEBUG_MODE:
                            if isinstance(history_data, dict):
                                print(f"History data keys: {list(history_data.keys())}")
                        
                        # Check if the response contains candles data
                        if isinstance(history_data, dict) and 'candles' in history_data:
                            candles = history_data['candles']
                            if candles:
                                # Convert to DataFrame
                                df = pd.DataFrame(candles)
                                
                                # Convert datetime column
                                if 'datetime' in df.columns:
                                    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                                
                                return df
                            else:
                                if DEBUG_MODE:
                                    print(f"No candles data found for {symbol}")
                        else:
                            if DEBUG_MODE:
                                print(f"No 'candles' key found in history data for {symbol}")
                    except Exception as e:
                        print(f"Error processing history data for {symbol}: {str(e)}")
                        if DEBUG_MODE:
                            print(f"Exception type: {type(e)}")
                            print(f"Traceback: {traceback.format_exc()}")
                elif isinstance(history, dict):
                    # If the client already returned a dict
                    if 'candles' in history:
                        candles = history['candles']
                        if candles:
                            # Convert to DataFrame
                            df = pd.DataFrame(candles)
                            
                            # Convert datetime column
                            if 'datetime' in df.columns:
                                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                            
                            return df
                        else:
                            if DEBUG_MODE:
                                print(f"No candles data found for {symbol}")
                    else:
                        if DEBUG_MODE:
                            print(f"No 'candles' key found in history data for {symbol}")
            
            # Return empty DataFrame if we couldn't get historical data
            return pd.DataFrame()
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_quote(self, symbol):
        """
        Get quote data for a symbol
        
        Args:
            symbol (str): The stock symbol to get quote for
            
        Returns:
            dict: Quote data
        """
        try:
            if DEBUG_MODE:
                print(f"Requesting quote for symbol: {symbol}")
            
            # Get quote data
            quote_response = self.client.quote(symbol)
            
            if DEBUG_MODE:
                print(f"Quote response type: {type(quote_response)}")
                if hasattr(quote_response, 'status_code'):
                    print(f"Status code: {quote_response.status_code}")
            
            # Process the response
            quote_data = None
            if isinstance(quote_response, requests.Response):
                if quote_response.status_code == 200:
                    try:
                        quote_data = quote_response.json()
                        if DEBUG_MODE:
                            print(f"Quote received for {symbol}")
                            if VERBOSE_DEBUG:
                                print(f"Quote data keys: {list(quote_data.keys() if isinstance(quote_data, dict) else [])}")
                        
                        # The Schwab API returns data in a specific format
                        # The actual quote data might be nested under the symbol key
                        if symbol in quote_data:
                            return quote_data[symbol]
                        else:
                            return quote_data
                    except ValueError as e:
                        print(f"Error parsing JSON for {symbol}: {str(e)}")
                        if DEBUG_MODE and hasattr(quote_response, 'text'):
                            print(f"Response text: {quote_response.text[:200]}...")
                else:
                    print(f"Quote response not OK for {symbol}. Status code: {quote_response.status_code}")
            elif isinstance(quote_response, dict):
                # If the client already returned a dict instead of a Response object
                if DEBUG_MODE:
                    print(f"Quote response is already a dict for {symbol}")
                
                if symbol in quote_response:
                    return quote_response[symbol]
                else:
                    return quote_response
            else:
                print(f"Unexpected quote response type for {symbol}: {type(quote_response)}")
            
            return None
        except Exception as e:
            print(f"Error retrieving quote for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_price_history(self, symbol, period_type='day', period=10, frequency_type='minute', 
                         frequency=30, need_extended_hours_data=True):
        """
        Get price history data for a symbol
        
        Args:
            symbol (str): The stock symbol
            period_type (str): Type of period - 'day', 'month', 'year', 'ytd'
            period (int): Number of periods
            frequency_type (str): Type of frequency - 'minute', 'daily', 'weekly', 'monthly'
            frequency (int): Frequency
            need_extended_hours_data (bool): Whether to include extended hours data
            
        Returns:
            pd.DataFrame: Price history data
        """
        try:
            if DEBUG_MODE:
                print(f"\n=== PRICE HISTORY REQUEST ===")
                print(f"Symbol: {symbol}")
                print(f"Parameters: periodType={period_type}, period={period}, frequencyType={frequency_type}, frequency={frequency}")
            
            # Use the existing get_historical_data method which already has all the necessary logic
            return self.get_historical_data(
                symbol=symbol,
                period_type=period_type,
                period=period,
                frequency_type=frequency_type,
                frequency=frequency,
                need_extended_hours_data=need_extended_hours_data
            )
        except Exception as e:
            print(f"Error retrieving price history for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_option_chain_with_underlying_price(self, symbol):
        """
        Get the option chain for a symbol with underlying price
        
        Args:
            symbol (str): The stock symbol to get options for
            
        Returns:
            dict: Option chain data with underlying price
        """
        try:
            if DEBUG_MODE:
                print(f"Requesting option chain with underlying price for: {symbol}")
            
            # Get option chain directly from client instead of calling get_option_chain to avoid circular reference
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
            if isinstance(option_chain_response, requests.Response):
                if option_chain_response.status_code == 200:
                    try:
                        option_chain = option_chain_response.json()
                        if DEBUG_MODE:
                            print(f"Option chain received for {symbol}, keys: {list(option_chain.keys() if isinstance(option_chain, dict) else [])}")
                    except ValueError as e:
                        print(f"Error parsing JSON for option chain: {str(e)}")
                        return None
                else:
                    print(f"Option chain response not OK. Status code: {option_chain_response.status_code}")
                    return None
            elif isinstance(option_chain_response, dict):
                option_chain = option_chain_response
            else:
                print(f"Unexpected option chain response type: {type(option_chain_response)}")
                return None
            
            if option_chain:
                # Extract underlying price
                underlying_price = None
                
                # Try to get underlying price from option chain
                if 'underlyingPrice' in option_chain:
                    underlying_price = option_chain['underlyingPrice']
                    if DEBUG_MODE:
                        print(f"underlyingPrice (camelCase): {underlying_price}")
                
                # If not found, try to get from underlying field
                if not underlying_price and 'underlying' in option_chain:
                    underlying = option_chain['underlying']
                    if DEBUG_MODE:
                        print(f"underlying field: {underlying}")
                        if isinstance(underlying, dict):
                            print(f"underlying field keys: {list(underlying.keys())}")
                    
                    if isinstance(underlying, dict):
                        # Try different possible price fields
                        for price_field in ['mark', 'last', 'close', 'bid', 'ask']:
                            if price_field in underlying:
                                underlying_price = underlying[price_field]
                                if DEBUG_MODE:
                                    print(f"underlying.{price_field}: {underlying_price}")
                                break
                
                # If still not found, get from quote
                if not underlying_price:
                    try:
                        quote = self.client.quote(symbol)
                        
                        if isinstance(quote, requests.Response) and quote.status_code == 200:
                            quote_data = quote.json()
                            
                            if symbol in quote_data:
                                symbol_data = quote_data[symbol]
                                underlying_price = symbol_data.get('lastPrice', 0)
                            else:
                                underlying_price = quote_data.get('lastPrice', 0)
                        elif isinstance(quote, dict):
                            if symbol in quote:
                                symbol_data = quote[symbol]
                                underlying_price = symbol_data.get('lastPrice', 0)
                            else:
                                underlying_price = quote.get('lastPrice', 0)
                    except Exception as e:
                        print(f"Error getting quote for underlying price: {str(e)}")
                        if DEBUG_MODE:
                            print(traceback.format_exc())
                
                # Use a default if all else fails
                if not underlying_price:
                    underlying_price = 0
                
                if DEBUG_MODE:
                    print(f"Using underlying price from option chain 'underlyingPrice': {underlying_price}")
                
                # Return option chain with underlying price
                return {
                    'symbol': symbol,
                    'underlying_price': underlying_price,
                    'option_chain': option_chain
                }
            else:
                if DEBUG_MODE:
                    print(f"No option chain data available for {symbol}")
                
                # Return empty data
                return {
                    'symbol': symbol,
                    'underlying_price': 0,
                    'option_chain': None
                }
        except Exception as e:
            print(f"Error retrieving option chain with underlying price for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            
            # Return empty data
            return {
                'symbol': symbol,
                'underlying_price': 0,
                'option_chain': None
            }
