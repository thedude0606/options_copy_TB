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
            frequency_type (str): The type of frequency (minute, daily, weekly, monthly)
            frequency (int): The frequency value
            lookback_days (int): Number of days to look back (alternative to period)
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            if DEBUG_MODE:
                print(f"Getting price data for {symbol}")
                print(f"Parameters: period_type={period_type}, period={period}, frequency_type={frequency_type}, frequency={frequency}")
            
            # Get the underlying symbol if this is an option
            underlying_symbol = self.get_underlying_symbol(symbol)
            
            # Use get_historical_data to fetch the data
            df = self.get_historical_data(
                symbol=underlying_symbol,
                period_type=period_type,
                period=period,
                frequency_type=frequency_type,
                frequency=frequency
            )
            
            if DEBUG_MODE:
                print(f"Price data received for {underlying_symbol}: {len(df)} rows")
                if not df.empty:
                    print(f"Price data columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving price data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1):
        """
        Get historical data for a symbol using Yahoo Finance API as a replacement for Schwab API
        
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
                print(f"Getting historical data for {symbol} using Yahoo Finance API")
                print(f"Parameters: period_type={period_type}, period={period}, frequency_type={frequency_type}, frequency={frequency}")
            
            # Get the underlying symbol if this is an option
            underlying_symbol = self.get_underlying_symbol(symbol)
            
            # Map Schwab API parameters to Yahoo Finance API parameters
            # Convert period_type and period to range
            range_param = '1mo'  # Default
            if period_type == 'day':
                if period <= 5:
                    range_param = '5d'
                else:
                    range_param = '1mo'
            elif period_type == 'month':
                if period <= 1:
                    range_param = '1mo'
                elif period <= 3:
                    range_param = '3mo'
                elif period <= 6:
                    range_param = '6mo'
                else:
                    range_param = '1y'
            elif period_type == 'year':
                if period <= 1:
                    range_param = '1y'
                elif period <= 2:
                    range_param = '2y'
                elif period <= 5:
                    range_param = '5y'
                else:
                    range_param = '10y'
            elif period_type == 'ytd':
                range_param = 'ytd'
                
            # Convert frequency_type and frequency to interval
            interval_param = '1d'  # Default
            if frequency_type == 'minute':
                if frequency == 1:
                    interval_param = '1m'
                elif frequency == 2:
                    interval_param = '2m'
                elif frequency == 5:
                    interval_param = '5m'
                elif frequency == 15:
                    interval_param = '15m'
                elif frequency == 30:
                    interval_param = '30m'
                elif frequency == 60:
                    interval_param = '60m'
                else:
                    interval_param = '1h'
            elif frequency_type == 'daily':
                interval_param = '1d'
            elif frequency_type == 'weekly':
                interval_param = '1wk'
            elif frequency_type == 'monthly':
                interval_param = '1mo'
                
            # Use Yahoo Finance API to get historical data
            # Create a temporary Python file to use the YahooFinance API
            temp_script_path = '/tmp/yahoo_finance_data.py'
            with open(temp_script_path, 'w') as f:
                f.write("""
import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import json

client = ApiClient()
symbol = sys.argv[1]
interval = sys.argv[2]
range_param = sys.argv[3]

# Call Yahoo Finance API
result = client.call_api('YahooFinance/get_stock_chart', query={
    'symbol': symbol,
    'interval': interval,
    'range': range_param,
    'includeAdjustedClose': True
})

# Process the result
if result and 'chart' in result and 'result' in result['chart'] and len(result['chart']['result']) > 0:
    data = result['chart']['result'][0]
    
    # Extract timestamp and indicators
    timestamps = data.get('timestamp', [])
    quote_data = data.get('indicators', {}).get('quote', [{}])[0]
    adjclose_data = data.get('indicators', {}).get('adjclose', [{}])[0].get('adjclose', [])
    
    # Create lists for each data point
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    adjcloses = []
    
    # Process each data point
    for i in range(len(timestamps)):
        if i < len(timestamps):
            dates.append(timestamps[i])
        else:
            dates.append(None)
            
        if 'open' in quote_data and i < len(quote_data['open']):
            opens.append(quote_data['open'][i])
        else:
            opens.append(None)
            
        if 'high' in quote_data and i < len(quote_data['high']):
            highs.append(quote_data['high'][i])
        else:
            highs.append(None)
            
        if 'low' in quote_data and i < len(quote_data['low']):
            lows.append(quote_data['low'][i])
        else:
            lows.append(None)
            
        if 'close' in quote_data and i < len(quote_data['close']):
            closes.append(quote_data['close'][i])
        else:
            closes.append(None)
            
        if 'volume' in quote_data and i < len(quote_data['volume']):
            volumes.append(quote_data['volume'][i])
        else:
            volumes.append(None)
            
        if i < len(adjclose_data):
            adjcloses.append(adjclose_data[i])
        else:
            adjcloses.append(None)
    
    # Create a dictionary with the data
    output_data = {
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'adjclose': adjcloses
    }
    
    # Output as JSON
    print(json.dumps(output_data))
else:
    print(json.dumps({}))
""")
            
            # Execute the script to get data from Yahoo Finance
            import subprocess
            cmd = f"python3 {temp_script_path} {underlying_symbol} {interval_param} {range_param}"
            if DEBUG_MODE:
                print(f"Executing command: {cmd}")
                
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if stderr and DEBUG_MODE:
                print(f"Error from Yahoo Finance API script: {stderr.decode('utf-8')}")
                
            if stdout:
                try:
                    # Parse the JSON output
                    history_data = json.loads(stdout.decode('utf-8'))
                    
                    if history_data and 'timestamp' in history_data:
                        # Create DataFrame
                        df = pd.DataFrame(history_data)
                        
                        # Convert timestamp to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        
                        if DEBUG_MODE:
                            print(f"Processed historical data: {len(df)} rows")
                            if not df.empty:
                                print(f"Historical data columns: {df.columns.tolist()}")
                                print(f"Historical data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                        
                        return df
                    else:
                        if DEBUG_MODE:
                            print(f"No data found in Yahoo Finance API response for {underlying_symbol}")
                        return pd.DataFrame()
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error parsing Yahoo Finance API response: {str(e)}")
                        print(f"Response: {stdout.decode('utf-8')[:500]}...")
                    return pd.DataFrame()
            else:
                if DEBUG_MODE:
                    print(f"No response from Yahoo Finance API for {underlying_symbol}")
                return pd.DataFrame()
            
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
            quote_data = None
            if hasattr(quote_response, 'json'):
                try:
                    quote_data = quote_response.json()
                    if DEBUG_MODE:
                        print(f"Quote received for {underlying_symbol}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error parsing quote JSON: {str(e)}")
                        if hasattr(quote_response, 'text'):
                            print(f"Response text: {quote_response.text[:500]}...")
            elif isinstance(quote_response, dict):
                quote_data = quote_response
                if DEBUG_MODE:
                    print(f"Quote received for {underlying_symbol}")
            else:
                if DEBUG_MODE:
                    print(f"No quote data received for {underlying_symbol}")
            
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
            market (str): The market to get hours for (EQUITY, OPTION, BOND, FOREX, FUTURES)
            
        Returns:
            dict: Market hours data
        """
        try:
            if DEBUG_MODE:
                print(f"Getting market hours for {market}")
            
            # Get market hours data
            hours_response = self.client.get_market_hours(
                markets=[market]
            )
            
            if DEBUG_MODE:
                print(f"Market hours response type: {type(hours_response)}")
                if hasattr(hours_response, 'status_code'):
                    print(f"Status code: {hours_response.status_code}")
            
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
                        if hasattr(hours_response, 'text'):
                            print(f"Response text: {hours_response.text[:500]}...")
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
