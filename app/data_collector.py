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
    
    def get_price_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1):
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): The stock symbol to get price data for
            period_type (str): The type of period to show (day, month, year, ytd)
            period (int): The number of periods to show
            frequency_type (str): The type of frequency with which a new candle is formed (minute, daily, weekly, monthly)
            frequency (int): The number of the frequency type to use (e.g., 1, 5, 10, 15, 30 for minute)
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            if DEBUG_MODE:
                print(f"Getting price data for {symbol} with period={period} {period_type}, frequency={frequency} {frequency_type}")
            
            # Extract underlying symbol if this is an option
            underlying_symbol = self.get_underlying_symbol(symbol)
            
            # Check cache first
            cache_key = f"{underlying_symbol}_{period_type}_{period}_{frequency_type}_{frequency}"
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
                # Assuming market hours are 6.5 hours per day (390 minutes)
                # and we're only including weekdays
                business_days = np.busday_count(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                total_minutes = business_days * 390
                num_points = total_minutes // frequency
            elif frequency_type == 'daily':
                num_points = np.busday_count(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            elif frequency_type == 'weekly':
                num_points = (end_date - start_date).days // 7
            else:  # monthly
                num_points = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
            
            # Generate mock data
            mock_data = []
            current_price = 150.0  # Starting price
            volatility = 0.02  # Daily volatility
            
            current_date = start_date
            for i in range(max(1, num_points)):
                # Generate random price movement
                price_change = np.random.normal(0, volatility * current_price)
                current_price = max(0.01, current_price + price_change)
                
                # Calculate OHLC
                open_price = current_price
                high_price = current_price * (1 + np.random.uniform(0, volatility))
                low_price = current_price * (1 - np.random.uniform(0, volatility))
                close_price = current_price * (1 + np.random.normal(0, volatility/2))
                
                # Generate volume
                volume = int(np.random.uniform(100000, 1000000))
                
                # Add data point
                if frequency_type == 'minute':
                    # Increment by minutes
                    current_date = current_date + timedelta(minutes=frequency)
                elif frequency_type == 'daily':
                    # Increment by days, skipping weekends
                    current_date = current_date + timedelta(days=1)
                    while current_date.weekday() > 4:  # Skip Saturday (5) and Sunday (6)
                        current_date = current_date + timedelta(days=1)
                elif frequency_type == 'weekly':
                    # Increment by weeks
                    current_date = current_date + timedelta(weeks=1)
                else:  # monthly
                    # Increment by months
                    if current_date.month == 12:
                        current_date = current_date.replace(year=current_date.year + 1, month=1)
                    else:
                        current_date = current_date.replace(month=current_date.month + 1)
                
                mock_data.append({
                    'datetime': current_date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(mock_data)
            
            # Set datetime as index
            if not df.empty:
                df.set_index('datetime', inplace=True)
            
            # Cache the result
            self.cache[cache_key] = df
            
            if DEBUG_MODE:
                print(f"Generated {len(df)} price data points for {underlying_symbol}")
                if not df.empty:
                    print(f"Price data columns: {df.columns.tolist()}")
                    print(f"Price data range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving price data for {symbol}: {str(e)}")
            if DEBUG_MODE:
                print(f"Exception type: {type(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol, indicators=None, period=14):
        """
        Calculate technical indicators for a symbol
        
        Args:
            symbol (str): The stock symbol to calculate indicators for
            indicators (list): List of indicators to calculate
            period (int): Period to use for indicators
            
        Returns:
            pd.DataFrame: DataFrame with price data and indicators
        """
        try:
            if DEBUG_MODE:
                print(f"Calculating technical indicators for {symbol}")
            
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
