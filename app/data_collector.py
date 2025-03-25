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
            quote = self.client.quote(symbol)
            
            if DEBUG_MODE:
                if quote:
                    print(f"Quote received for {symbol}")
                    if hasattr(quote, 'json'):
                        try:
                            quote_data = quote.json()
                            print(f"Quote data keys: {list(quote_data.keys() if isinstance(quote_data, dict) else [])}")
                        except:
                            print("Could not parse quote JSON")
                else:
                    print(f"No quote data received for {symbol}")
            
            return quote
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
            
            # Get market hours
            hours = self.client.get_market_hours(market=market)
            
            if DEBUG_MODE:
                if hours:
                    print(f"Market hours received for {market}")
                else:
                    print(f"No market hours data received for {market}")
            
            return hours
        except Exception as e:
            print(f"Error retrieving market hours: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_option_data(self, symbol, option_type='ALL', strike=None, expiration=None):
        """
        Get detailed options data including Greeks
        
        Args:
            symbol (str): The stock symbol to get options for
            option_type (str): Option type - 'CALL', 'PUT', or 'ALL'
            strike (float): Specific strike price to filter by
            expiration (str): Specific expiration date to filter by (format: 'YYYY-MM-DD')
            
        Returns:
            pd.DataFrame: Options data with Greeks
        """
        try:
            if DEBUG_MODE:
                print(f"\n=== OPTIONS DATA REQUEST ===")
                print(f"Symbol: {symbol}")
                print(f"Parameters: option_type={option_type}, strike={strike}, expiration={expiration}")
            
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                print(f"Error: Invalid symbol: {symbol}")
                return pd.DataFrame()
                
            # Get option chain
            option_chain = self.get_option_chain(symbol)
            
            # Enhanced error handling and debugging for option chain
            if not option_chain:
                print(f"No options data available for {symbol}")
                return pd.DataFrame()
                
            if VERBOSE_DEBUG:
                print(f"Option chain type: {type(option_chain)}")
                if isinstance(option_chain, dict):
                    print(f"Option chain keys: {list(option_chain.keys())}")
                    
                    # Check for expected keys
                    expected_keys = ['callExpDateMap', 'putExpDateMap', 'underlyingPrice']
                    missing_keys = [key for key in expected_keys if key not in option_chain]
                    if missing_keys:
                        print(f"Warning: Missing expected keys in option chain: {missing_keys}")
                        
                    # Log underlying price if available
                    if 'underlyingPrice' in option_chain:
                        print(f"Underlying price: {option_chain['underlyingPrice']}")
                    elif 'underlying' in option_chain and 'mark' in option_chain['underlying']:
                        print(f"Underlying price from 'underlying.mark': {option_chain['underlying']['mark']}")
                    else:
                        print("Warning: No underlying price found in option chain")
            
            # Extract options data
            options_data = []
            
            # Process call options with enhanced error handling
            if option_type in ['CALL', 'ALL'] and 'callExpDateMap' in option_chain:
                if DEBUG_MODE:
                    print(f"Processing call options, expiration dates: {len(option_chain['callExpDateMap'])}")
                
                try:
                    for exp_date, strikes in option_chain['callExpDateMap'].items():
                        # Skip if not matching expiration filter
                        if expiration and expiration not in exp_date:
                            continue
                        
                        if not isinstance(strikes, dict):
                            if DEBUG_MODE:
                                print(f"Warning: Strikes is not a dictionary for expiration {exp_date}, type: {type(strikes)}")
                            continue
                            
                        for strike_price, options in strikes.items():
                            # Skip if not matching strike filter
                            try:
                                if strike and float(strike_price) != float(strike):
                                    continue
                                    
                                if not isinstance(options, list):
                                    if DEBUG_MODE:
                                        print(f"Warning: Options is not a list for strike {strike_price}, type: {type(options)}")
                                    continue
                                
                                for option in options:
                                    try:
                                        option['optionType'] = 'CALL'
                                        option['expirationDate'] = exp_date.split(':')[0]
                                        option['strikePrice'] = float(strike_price)
                                        options_data.append(option)
                                    except Exception as e:
                                        if DEBUG_MODE:
                                            print(f"Error processing call option: {str(e)}")
                                            print(f"Option data: {option}")
                            except ValueError as e:
                                if DEBUG_MODE:
                                    print(f"Error converting strike price: {str(e)}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error processing call options: {str(e)}")
                        print(traceback.format_exc())
            
            # Process put options with enhanced error handling
            if option_type in ['PUT', 'ALL'] and 'putExpDateMap' in option_chain:
                if DEBUG_MODE:
                    print(f"Processing put options, expiration dates: {len(option_chain['putExpDateMap'])}")
                
                try:
                    for exp_date, strikes in option_chain['putExpDateMap'].items():
                        # Skip if not matching expiration filter
                        if expiration and expiration not in exp_date:
                            continue
                        
                        if not isinstance(strikes, dict):
                            if DEBUG_MODE:
                                print(f"Warning: Strikes is not a dictionary for expiration {exp_date}, type: {type(strikes)}")
                            continue
                            
                        for strike_price, options in strikes.items():
                            # Skip if not matching strike filter
                            try:
                                if strike and float(strike_price) != float(strike):
                                    continue
                                    
                                if not isinstance(options, list):
                                    if DEBUG_MODE:
                                        print(f"Warning: Options is not a list for strike {strike_price}, type: {type(options)}")
                                    continue
                                
                                for option in options:
                                    try:
                                        option['optionType'] = 'PUT'
                                        option['expirationDate'] = exp_date.split(':')[0]
                                        option['strikePrice'] = float(strike_price)
                                        options_data.append(option)
                                    except Exception as e:
                                        if DEBUG_MODE:
                                            print(f"Error processing put option: {str(e)}")
                                            print(f"Option data: {option}")
                            except ValueError as e:
                                if DEBUG_MODE:
                                    print(f"Error converting strike price: {str(e)}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error processing put options: {str(e)}")
                        print(traceback.format_exc())
            
            # Convert to DataFrame with enhanced error handling
            if options_data:
                if DEBUG_MODE:
                    print(f"Found {len(options_data)} options")
                
                try:
                    df = pd.DataFrame(options_data)
                    
                    # Add underlying price to all options with fallback mechanisms
                    underlying_price = None
                    
                    # Try different ways to get the underlying price
                    if 'underlyingPrice' in option_chain:
                        underlying_price = option_chain['underlyingPrice']
                        if DEBUG_MODE:
                            print(f"Using underlying price from 'underlyingPrice': {underlying_price}")
                    elif 'underlying' in option_chain and isinstance(option_chain['underlying'], dict):
                        if 'mark' in option_chain['underlying']:
                            underlying_price = option_chain['underlying']['mark']
                            if DEBUG_MODE:
                                print(f"Using underlying price from 'underlying.mark': {underlying_price}")
                        elif 'last' in option_chain['underlying']:
                            underlying_price = option_chain['underlying']['last']
                            if DEBUG_MODE:
                                print(f"Using underlying price from 'underlying.last': {underlying_price}")
                    
                    # If we still don't have a price, try to get a quote
                    if underlying_price is None:
                        if DEBUG_MODE:
                            print(f"No underlying price in option chain, trying to get quote")
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
                    
                    # Set the underlying price in the DataFrame
                    if underlying_price is not None:
                        df['underlyingPrice'] = underlying_price
                    else:
                        if DEBUG_MODE:
                            print(f"Warning: Could not determine underlying price")
                        # Set a placeholder value to avoid errors
                        df['underlyingPrice'] = 0.0
                        
                    # Convert expiration date to datetime with error handling
                    if 'expirationDate' in df.columns:
                        try:
                            df['expirationDate'] = pd.to_datetime(df['expirationDate'])
                        except Exception as e:
                            if DEBUG_MODE:
                                print(f"Error converting expiration dates: {str(e)}")
                                print(f"Expiration date values: {df['expirationDate'].unique()}")
                            # Try a different format or set to NaT
                            try:
                                df['expirationDate'] = pd.to_datetime(df['expirationDate'], format='%Y-%m-%d')
                            except:
                                df['expirationDate'] = pd.NaT
                    
                    # Calculate days to expiration with error handling
                    if 'expirationDate' in df.columns:
                        try:
                            df['daysToExpiration'] = (df['expirationDate'] - pd.Timestamp.now())
                            
                            # Create a numeric days column to avoid .dt accessor issues
                            def get_days(x):
                                try:
                                    if isinstance(x, pd.Timedelta):
                                        return x.days
                                    elif pd.isna(x):
                                        return 0
                                    else:
                                        return float(x)
                                except:
                                    return 0
                            
                            # Apply the function to create a numeric days column
                            df['days_numeric'] = df['daysToExpiration'].apply(get_days)
                        except Exception as e:
                            if DEBUG_MODE:
                                print(f"Error calculating days to expiration: {str(e)}")
                            # Set default values
                            df['days_numeric'] = 0
                    
                    if DEBUG_MODE:
                        print(f"DataFrame columns: {list(df.columns)}")
                        print(f"DataFrame shape: {df.shape}")
                    
                    if VERBOSE_DEBUG:
                        # Print sample data for debugging
                        print("\nSample data (first 2 rows):")
                        if len(df) > 0:
                            print(df.head(2).to_string())
                        
                        # Check for missing critical columns
                        critical_columns = ['strikePrice', 'underlyingPrice', 'expirationDate', 'optionType']
                        missing_columns = [col for col in critical_columns if col not in df.columns]
                        if missing_columns:
                            print(f"Warning: Missing critical columns: {missing_columns}")
                    
                    return df
                except Exception as e:
                    print(f"Error creating DataFrame from options data: {str(e)}")
                    if DEBUG_MODE:
                        print(traceback.format_exc())
                    return pd.DataFrame()
            else:
                print(f"No options data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error retrieving options data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_put_call_ratio(self, symbol):
        """
        Calculate put/call ratio for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            float: Put/Call ratio
        """
        try:
            if DEBUG_MODE:
                print(f"Calculating put/call ratio for: {symbol}")
            
            # Get option chain
            option_chain = self.get_option_chain(symbol)
            if not option_chain:
                return None
            
            # Calculate total volume for calls and puts
            call_volume = 0
            put_volume = 0
            
            # Process call options
            if 'callExpDateMap' in option_chain:
                for exp_date, strikes in option_chain['callExpDateMap'].items():
                    for strike_price, options in strikes.items():
                        for option in options:
                            if 'totalVolume' in option:
                                call_volume += option['totalVolume']
            
            # Process put options
            if 'putExpDateMap' in option_chain:
                for exp_date, strikes in option_chain['putExpDateMap'].items():
                    for strike_price, options in strikes.items():
                        for option in options:
                            if 'totalVolume' in option:
                                put_volume += option['totalVolume']
            
            if DEBUG_MODE:
                print(f"Call volume: {call_volume}, Put volume: {put_volume}")
            
            # Calculate ratio
            if call_volume > 0:
                ratio = put_volume / call_volume
                if DEBUG_MODE:
                    print(f"Put/Call ratio: {ratio}")
                return ratio
            else:
                if DEBUG_MODE:
                    print("Call volume is zero, cannot calculate ratio")
                return None
                
        except Exception as e:
            print(f"Error calculating put/call ratio for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_open_interest(self, symbol, option_type='ALL'):
        """
        Get open interest data for a symbol
        
        Args:
            symbol (str): The stock symbol
            option_type (str): Option type - 'CALL', 'PUT', or 'ALL'
            
        Returns:
            pd.DataFrame: Open interest data
        """
        try:
            if DEBUG_MODE:
                print(f"Getting open interest data for: {symbol}, type: {option_type}")
            
            # Get option data
            options_df = self.get_option_data(symbol, option_type=option_type)
            if options_df.empty:
                return pd.DataFrame()
            
            # Extract open interest data
            if 'openInterest' in options_df.columns:
                oi_data = options_df[['strikePrice', 'expirationDate', 'optionType', 'openInterest']]
                
                if DEBUG_MODE:
                    print(f"Open interest data shape: {oi_data.shape}")
                
                return oi_data
            else:
                if DEBUG_MODE:
                    print("No openInterest column found in options data")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error retrieving open interest data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_streaming_data(self, symbols, fields=None):
        """
        Set up streaming data for symbols
        
        Args:
            symbols (list): List of symbols to stream
            fields (list): List of fields to stream
            
        Returns:
            object: Streaming data handler
        """
        try:
            if DEBUG_MODE:
                print(f"Setting up streaming data for symbols: {symbols}")
            
            # Default fields if none provided
            if not fields:
                fields = [
                    'LAST_PRICE', 'BID_PRICE', 'ASK_PRICE', 'TOTAL_VOLUME',
                    'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE',
                    'NET_CHANGE', 'VOLATILITY', 'DELTA', 'GAMMA', 'THETA', 'VEGA'
                ]
            
            # Initialize streamer
            streamer = self.client.stream
            
            if DEBUG_MODE:
                print(f"Streamer initialized: {type(streamer)}")
            
            # Return streamer for further configuration
            return streamer
                
        except Exception as e:
            print(f"Error setting up streaming data: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return None
