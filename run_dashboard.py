import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json

from app.data_collector import DataCollector
from app.components.header import create_header
from app.components.sidebar import create_sidebar
from app.components.dashboard_content import create_dashboard_content
from app.components.trade_card import create_trade_cards_container
from app.analysis.recommendation_engine import ShortTermRecommendationEngine
from app.analysis.options_analysis import OptionsAnalysis
from app.integration import register_callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

class OptionsDashboard:
    def __init__(self, interactive_auth=False):
        """
        Initialize the options dashboard application
        
        Args:
            interactive_auth (bool): Whether to use interactive authentication
        """
        # Initialize data collector
        logger.info("Initializing data collector...")
        self.data_collector = DataCollector(interactive_auth=interactive_auth)
        self.client = self.data_collector.client
        print(f"DataCollector initialized with interactive_auth={interactive_auth}")
        print(f"Client type: {type(self.client)}")
        logger.info("Data collector initialized successfully")
        
        # Initialize platform components
        logger.info("Initializing platform components...")
        self.recommendation_engine = ShortTermRecommendationEngine(self)
        self.options_analysis = OptionsAnalysis()
        logger.info("Platform components initialized successfully")
        
        # Create Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        self.app.title = "Options Recommendation Platform"
        
        # Create app layout
        logger.info("Creating app layout...")
        self.create_layout()
        logger.info("App layout created successfully")
        
        # Register callbacks
        logger.info("Registering callbacks...")
        register_callbacks(self.app, self)
        logger.info("Callbacks registered successfully")
    
    def create_layout(self):
        """Create the application layout"""
        self.app.layout = html.Div([
            dcc.Store(id="options-data-store"),
            dcc.Store(id="market-data-store"),
            dcc.Store(id="recommendations-store"),
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # in milliseconds (1 minute)
                n_intervals=0
            ),
            create_header(),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(create_sidebar(), width=2),
                            dbc.Col(create_dashboard_content(), width=10)
                        ]
                    )
                ],
                fluid=True,
                className="mt-4"
            )
        ])
    
    def get_option_chain(self, symbol):
        """
        Get option chain data for a symbol
        
        Args:
            symbol (str): The stock symbol to get options for
            
        Returns:
            dict: Option chain data
        """
        try:
            # Get the current price of the underlying
            quote_response = self.client.quote(symbol)
            
            # Enhanced error handling and debugging for quote data
            if hasattr(quote_response, 'json'):
                quote_data = quote_response.json()
                if 'lastPrice' in quote_data:
                    current_price = quote_data.get('lastPrice', 0)
                    if current_price <= 0:
                        logger.warning(f"Quote data for {symbol} has invalid lastPrice: {current_price}")
                        # Try to get price from underlying quote in option chain if available
                        current_price = 0
                else:
                    logger.warning(f"Quote data for {symbol} missing lastPrice field. Available fields: {list(quote_data.keys())}")
                    current_price = 0
            else:
                logger.warning(f"Quote response for {symbol} is not JSON serializable: {type(quote_response)}")
                current_price = 0
            
            # Get option chain data
            option_chain_response = self.client.option_chains(
                symbol=symbol,
                contractType="ALL",
                strikeCount=10,  # Get options around the current price
                includeUnderlyingQuote=True,
                strategy="SINGLE"
            )
            
            if hasattr(option_chain_response, 'json'):
                option_chain_data = option_chain_response.json()
                
                # If we couldn't get current price from quote, try to get it from option chain
                if current_price <= 0 and 'underlyingPrice' in option_chain_data:
                    current_price = option_chain_data.get('underlyingPrice', 0)
                    logger.info(f"Using underlyingPrice from option chain for {symbol}: {current_price}")
            else:
                option_chain_data = {}
            
            # Process the option chain data
            expiration_dates = []
            options = []
            
            # Extract expiration dates and options data
            for exp_date in option_chain_data.get('callExpDateMap', {}).keys():
                expiration_dates.append(exp_date.split(':')[0])
                
                # Process call options
                for strike in option_chain_data.get('callExpDateMap', {}).get(exp_date, {}):
                    for call_option in option_chain_data.get('callExpDateMap', {}).get(exp_date, {}).get(strike, []):
                        options.append({
                            "option_type": "CALL",
                            "symbol": call_option.get('symbol'),
                            "strike": float(strike),
                            "expiration": exp_date.split(':')[0],
                            "bid": call_option.get('bid', 0),
                            "ask": call_option.get('ask', 0),
                            "last": call_option.get('last', 0),
                            "volume": call_option.get('totalVolume', 0),
                            "open_interest": call_option.get('openInterest', 0),
                            "delta": call_option.get('delta', 0),
                            "gamma": call_option.get('gamma', 0),
                            "theta": call_option.get('theta', 0),
                            "vega": call_option.get('vega', 0),
                            "implied_volatility": call_option.get('volatility', 0) / 100  # Convert to decimal
                        })
                
                # Process put options
                for strike in option_chain_data.get('putExpDateMap', {}).get(exp_date, {}):
                    for put_option in option_chain_data.get('putExpDateMap', {}).get(exp_date, {}).get(strike, []):
                        options.append({
                            "option_type": "PUT",
                            "symbol": put_option.get('symbol'),
                            "strike": float(strike),
                            "expiration": exp_date.split(':')[0],
                            "bid": put_option.get('bid', 0),
                            "ask": put_option.get('ask', 0),
                            "last": put_option.get('last', 0),
                            "volume": put_option.get('totalVolume', 0),
                            "open_interest": put_option.get('openInterest', 0),
                            "delta": put_option.get('delta', 0),
                            "gamma": put_option.get('gamma', 0),
                            "theta": put_option.get('theta', 0),
                            "vega": put_option.get('vega', 0),
                            "implied_volatility": put_option.get('volatility', 0) / 100  # Convert to decimal
                        })
            
            return {
                "symbol": symbol,
                "underlying_price": current_price,
                "expirations": list(set(expiration_dates)),  # Remove duplicates
                "options": options
            }
        except Exception as e:
            logger.error(f"Error retrieving option chain for {symbol}: {str(e)}")
            # Return None or empty data structure
            return {
                "symbol": symbol,
                "underlying_price": 0,
                "expirations": [],
                "options": []
            }
    
    def get_market_data(self):
        """
        Get market data for dashboard
        
        Returns:
            dict: Market data
        """
        try:
            # Get quotes for major indices
            indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']
            quotes = {}
            
            for symbol in indices:
                try:
                    print(f"Requesting quote for symbol: {symbol}")
                    quote_response = self.client.quote(symbol)
                    
                    if hasattr(quote_response, 'json'):
                        quote_data = quote_response.json()
                        quotes[symbol] = {
                            'lastPrice': quote_data.get('lastPrice', 0),
                            'netChange': quote_data.get('netChange', 0),
                            'netPercentChangeInDouble': quote_data.get('netPercentChangeInDouble', 0),
                            'totalVolume': quote_data.get('totalVolume', 0),
                            'description': quote_data.get('description', symbol)
                        }
                        print(f"Quote received for {symbol}")
                        print(f"Quote data keys: {list(quotes.keys())}")
                    else:
                        print(f"No quote data received for {symbol}")
                        if hasattr(quote_response, 'status_code'):
                            print(f"Quote response not OK. Status code: {quote_response.status_code}")
                except Exception as e:
                    print(f"Error getting quote for {symbol}: {str(e)}")
            
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
            logger.error(f"Error retrieving market data: {str(e)}")
            return {
                'quotes': {},
                'date': datetime.now().strftime('%Y-%m-%d'),
                'market_movers': {'most_active': [], 'gainers': [], 'losers': []}
            }
    
    def run(self, debug=True, port=8050):
        """
        Run the dashboard application
        
        Args:
            debug (bool): Whether to run in debug mode
            port (int): Port to run the application on
        """
        logger.info("Starting application...")
        self.app.run_server(debug=debug, port=port)

if __name__ == '__main__':
    dashboard = OptionsDashboard(interactive_auth=False)
    dashboard.run(debug=True)
