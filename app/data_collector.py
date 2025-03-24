"""
Data collection module for options recommendation platform.
Handles retrieving market data from Schwab API for technical indicators and options analysis.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.auth import get_client

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
            # Get historical price data with retry logic
            history = None
            
            # Try with primary parameters
            try:
                history = self.client.price_history(
                    symbol=symbol,
                    period_type=period_type,
                    period=period,
                    frequency_type=frequency_type,
                    frequency=frequency,
                    need_extended_hours_data=need_extended_hours_data
                )
            except Exception as e:
                print(f"Primary parameters failed: {str(e)}")
            
            # If primary parameters failed, try alternative configurations
            if not history:
                try:
                    # Try with daily frequency
                    history = self.client.price_history(
                        symbol=symbol,
                        period_type='month',
                        period=1,
                        frequency_type='daily',
                        frequency=1,
                        need_extended_hours_data=need_extended_hours_data
                    )
                except Exception as e:
                    print(f"Alternative parameters failed: {str(e)}")
            
            # Process historical data
            if history and 'candles' in history:
                # Convert to DataFrame
                df = pd.DataFrame(history['candles'])
                
                # Convert datetime
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                
                return df
            else:
                print(f"No valid historical data returned for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error retrieving historical data for {symbol}: {str(e)}")
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
            # Get quote data
            quote = self.client.quote(symbol)
            return quote
        except Exception as e:
            print(f"Error retrieving quote for {symbol}: {str(e)}")
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
            # Get market hours
            hours = self.client.get_market_hours(market=market)
            return hours
        except Exception as e:
            print(f"Error retrieving market hours: {str(e)}")
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
            # Get option chain
            option_chain = self.get_option_chain(symbol)
            if not option_chain:
                return pd.DataFrame()
            
            # Extract options data
            options_data = []
            
            # Process call options
            if option_type in ['CALL', 'ALL'] and 'callExpDateMap' in option_chain:
                for exp_date, strikes in option_chain['callExpDateMap'].items():
                    # Skip if not matching expiration filter
                    if expiration and expiration not in exp_date:
                        continue
                    
                    for strike_price, options in strikes.items():
                        # Skip if not matching strike filter
                        if strike and float(strike_price) != float(strike):
                            continue
                        
                        for option in options:
                            option['optionType'] = 'CALL'
                            option['expirationDate'] = exp_date.split(':')[0]
                            option['strikePrice'] = float(strike_price)
                            options_data.append(option)
            
            # Process put options
            if option_type in ['PUT', 'ALL'] and 'putExpDateMap' in option_chain:
                for exp_date, strikes in option_chain['putExpDateMap'].items():
                    # Skip if not matching expiration filter
                    if expiration and expiration not in exp_date:
                        continue
                    
                    for strike_price, options in strikes.items():
                        # Skip if not matching strike filter
                        if strike and float(strike_price) != float(strike):
                            continue
                        
                        for option in options:
                            option['optionType'] = 'PUT'
                            option['expirationDate'] = exp_date.split(':')[0]
                            option['strikePrice'] = float(strike_price)
                            options_data.append(option)
            
            # Convert to DataFrame
            if options_data:
                df = pd.DataFrame(options_data)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error retrieving options data for {symbol}: {str(e)}")
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
            
            # Calculate ratio
            if call_volume > 0:
                return put_volume / call_volume
            else:
                return None
                
        except Exception as e:
            print(f"Error calculating put/call ratio for {symbol}: {str(e)}")
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
            # Get option data
            options_df = self.get_option_data(symbol, option_type=option_type)
            if options_df.empty:
                return pd.DataFrame()
            
            # Extract open interest data
            if 'openInterest' in options_df.columns:
                oi_data = options_df[['strikePrice', 'expirationDate', 'optionType', 'openInterest']]
                return oi_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error retrieving open interest data for {symbol}: {str(e)}")
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
            # Default fields if none provided
            if not fields:
                fields = [
                    'LAST_PRICE', 'BID_PRICE', 'ASK_PRICE', 'TOTAL_VOLUME',
                    'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE',
                    'NET_CHANGE', 'VOLATILITY', 'DELTA', 'GAMMA', 'THETA', 'VEGA'
                ]
            
            # Initialize streamer
            streamer = self.client.stream
            
            # Return streamer for further configuration
            return streamer
                
        except Exception as e:
            print(f"Error setting up streaming data: {str(e)}")
            return None
