import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from app.auth import get_client

class OptionsDataRetriever:
    """
    Class to retrieve options data from Schwab API
    """
    def __init__(self, interactive_auth=False):
        """
        Initialize the options data retriever
        
        Args:
            interactive_auth (bool): Whether to allow interactive authentication
        """
        self.client = get_client(interactive=interactive_auth)
    
    def get_option_chain(self, symbol):
        """
        Get the option chain for a symbol
        
        Args:
            symbol (str): The stock symbol to get options for
            
        Returns:
            dict: Option chain data
        """
        try:
            # Get option chain data
            option_chain = self.client.option_chains(symbol)
            return option_chain
        except Exception as e:
            print(f"Error retrieving option chain for {symbol}: {str(e)}")
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
            print(f"Getting option data for {symbol}, type={option_type}, strike={strike}, expiration={expiration}")
            
            # Get option chain
            option_chain = self.get_option_chain(symbol)
            if not option_chain:
                print(f"No option chain returned for {symbol}")
                return pd.DataFrame()
            
            # Debug: Print option chain structure
            print(f"Option chain type: {type(option_chain)}")
            if isinstance(option_chain, dict):
                print(f"Option chain keys: {list(option_chain.keys())}")
            
            # Process option chain data to extract Greeks and other details
            options_data = []
            
            # Check if we have a Response object or a dict
            if hasattr(option_chain, 'json'):
                try:
                    option_chain_data = option_chain.json()
                    print(f"Converted Response to JSON for {symbol}")
                except Exception as e:
                    print(f"Error converting Response to JSON: {str(e)}")
                    return pd.DataFrame()
            else:
                option_chain_data = option_chain
            
            # Check for callExpDateMap and putExpDateMap (standard Schwab API format)
            call_map = option_chain_data.get('callExpDateMap', {})
            put_map = option_chain_data.get('putExpDateMap', {})
            
            print(f"Call expirations: {list(call_map.keys())}")
            print(f"Put expirations: {list(put_map.keys())}")
            
            # Process call options
            if option_type in ['ALL', 'CALL']:
                for exp_date_str, strikes in call_map.items():
                    # Extract date part (format: "YYYY-MM-DD:X")
                    exp_date_parts = exp_date_str.split(':')
                    if len(exp_date_parts) > 0:
                        exp_date = exp_date_parts[0]
                    else:
                        exp_date = exp_date_str
                    
                    # Skip if not matching expiration filter
                    if expiration and exp_date != expiration:
                        continue
                    
                    for strike_price, strike_data in strikes.items():
                        # Skip if not matching strike filter
                        if strike and float(strike_price) != float(strike):
                            continue
                        
                        # Get the first (and usually only) option at this strike
                        if len(strike_data) > 0:
                            option = strike_data[0]
                            
                            # Extract option data including Greeks
                            option_info = {
                                'symbol': symbol,
                                'option_type': 'CALL',
                                'strike': float(strike_price),
                                'expiration': exp_date,
                                'bid': option.get('bid', 0),
                                'ask': option.get('ask', 0),
                                'last': option.get('last', 0),
                                'volume': option.get('totalVolume', 0),
                                'openInterest': option.get('openInterest', 0),
                                'delta': option.get('delta', 0),
                                'gamma': option.get('gamma', 0),
                                'theta': option.get('theta', 0),
                                'vega': option.get('vega', 0),
                                'rho': option.get('rho', 0),
                                'impliedVolatility': option.get('volatility', 0),
                            }
                            options_data.append(option_info)
            
            # Process put options
            if option_type in ['ALL', 'PUT']:
                for exp_date_str, strikes in put_map.items():
                    # Extract date part (format: "YYYY-MM-DD:X")
                    exp_date_parts = exp_date_str.split(':')
                    if len(exp_date_parts) > 0:
                        exp_date = exp_date_parts[0]
                    else:
                        exp_date = exp_date_str
                    
                    # Skip if not matching expiration filter
                    if expiration and exp_date != expiration:
                        continue
                    
                    for strike_price, strike_data in strikes.items():
                        # Skip if not matching strike filter
                        if strike and float(strike_price) != float(strike):
                            continue
                        
                        # Get the first (and usually only) option at this strike
                        if len(strike_data) > 0:
                            option = strike_data[0]
                            
                            # Extract option data including Greeks
                            option_info = {
                                'symbol': symbol,
                                'option_type': 'PUT',
                                'strike': float(strike_price),
                                'expiration': exp_date,
                                'bid': option.get('bid', 0),
                                'ask': option.get('ask', 0),
                                'last': option.get('last', 0),
                                'volume': option.get('totalVolume', 0),
                                'openInterest': option.get('openInterest', 0),
                                'delta': option.get('delta', 0),
                                'gamma': option.get('gamma', 0),
                                'theta': option.get('theta', 0),
                                'vega': option.get('vega', 0),
                                'rho': option.get('rho', 0),
                                'impliedVolatility': option.get('volatility', 0),
                            }
                            options_data.append(option_info)
            
            # Create DataFrame from collected options data
            df = pd.DataFrame(options_data)
            print(f"Processed {len(df)} options for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error retrieving options data for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    
    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1):
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): The stock symbol
            period_type (str): Type of period - 'day', 'month', 'year', 'ytd'
            period (int): Number of periods
            frequency_type (str): Type of frequency - 'minute', 'daily', 'weekly', 'monthly'
            frequency (int): Frequency
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            print(f"Getting historical data for {symbol}, period_type={period_type}, period={period}, frequency_type={frequency_type}, frequency={frequency}")
            
            # Get historical price data
            history = self.client.price_history(
                symbol=symbol,
                period_type=period_type,
                period=period,
                frequency_type=frequency_type,
                frequency=frequency
            )
            
            # Debug: Print history structure
            print(f"History response type: {type(history)}")
            
            # Process historical data
            historical_data = []
            
            # Check if we have a Response object or a dict
            if hasattr(history, 'json'):
                try:
                    history_data = history.json()
                    print(f"Converted Response to JSON for {symbol}")
                    if isinstance(history_data, dict):
                        print(f"History data keys: {list(history_data.keys())}")
                except Exception as e:
                    print(f"Error converting Response to JSON: {str(e)}")
                    return pd.DataFrame()
            else:
                history_data = history
                if isinstance(history_data, dict):
                    print(f"History data keys: {list(history_data.keys())}")
            
            # Check for candles in the response (standard Schwab API format)
            candles = None
            
            # Try different possible structures
            if isinstance(history_data, dict):
                # Try standard structure
                if 'candles' in history_data:
                    candles = history_data['candles']
                # Try nested structure
                elif 'price_history' in history_data and 'candles' in history_data['price_history']:
                    candles = history_data['price_history']['candles']
                # Try another common structure
                elif symbol in history_data and 'candles' in history_data[symbol]:
                    candles = history_data[symbol]['candles']
            
            if candles and isinstance(candles, list):
                print(f"Found {len(candles)} candles for {symbol}")
                
                for candle in candles:
                    # Extract timestamp and convert to datetime
                    timestamp = candle.get('datetime', 0)
                    if isinstance(timestamp, int):
                        # Convert milliseconds to datetime
                        dt = datetime.fromtimestamp(timestamp / 1000)
                    else:
                        # Try to parse as string
                        try:
                            dt = datetime.fromisoformat(str(timestamp))
                        except:
                            dt = datetime.now()  # Fallback
                    
                    # Extract OHLCV data
                    candle_data = {
                        'datetime': dt,
                        'open': candle.get('open', 0),
                        'high': candle.get('high', 0),
                        'low': candle.get('low', 0),
                        'close': candle.get('close', 0),
                        'volume': candle.get('volume', 0)
                    }
                    historical_data.append(candle_data)
            else:
                print(f"No candles found in history data for {symbol}")
                # If we can't find candles in the expected format, dump the structure for debugging
                if isinstance(history_data, dict):
                    print("History data structure:")
                    for key, value in history_data.items():
                        print(f"  {key}: {type(value)}")
                        if isinstance(value, dict):
                            print(f"    Keys: {list(value.keys())}")
            
            # Create DataFrame from collected historical data
            df = pd.DataFrame(historical_data)
            if not df.empty and 'datetime' in df.columns:
                # Sort by datetime
                df = df.sort_values('datetime')
            
            print(f"Processed {len(df)} historical data points for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
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
            print(f"Getting quote for {symbol}")
            
            # Get quote data
            quote_response = self.client.quote(symbol)
            
            # Debug: Print quote response structure
            print(f"Quote response type: {type(quote_response)}")
            
            # Process the response
            if isinstance(quote_response, requests.Response):
                if quote_response.status_code == 200:
                    try:
                        quote_data = quote_response.json()
                        print(f"Quote JSON data for {symbol}: {quote_data}")
                        return quote_data
                    except Exception as e:
                        print(f"Error parsing quote JSON: {str(e)}")
                        return None
                else:
                    print(f"Quote response not OK for {symbol}. Status code: {quote_response.status_code}")
                    return None
            else:
                # If it's already a dict or other data structure
                print(f"Quote response is already processed for {symbol}")
                return quote_response
                
        except Exception as e:
            print(f"Error retrieving quote for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
