"""
Full-featured Schwab Options Dashboard application with real API data integration
"""
import os
import sys
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
from dotenv import load_dotenv
import requests
import json
from datetime import datetime, timedelta
import time
import webbrowser
import random
import math
import dash_bootstrap_components as dbc

# Import custom modules
from app.streaming_data import StreamingDataManager
from app.stream_data_handler import StreamDataHandler
from app.real_time_tab import register_real_time_callbacks
from app.components.indicators_tab import register_indicators_callbacks, create_indicators_tab
from app.components.greeks_tab import register_greeks_callbacks
from app.historical_tab import register_historical_callbacks
from app.components.recommendations_tab import create_recommendations_tab, register_recommendations_callbacks
from app.components.trade_card import create_trade_cards_container
from app.analysis.recommendation_engine import RecommendationEngine
from app.data_collector import DataCollector

# Import debugging tools for timeline selector
from tests.debug_timeline_selector import add_debug_callback_to_app, add_debug_div_to_layout

# Load environment variables
load_dotenv()

# Authentication and API credentials
SCHWAB_APP_KEY = os.getenv('SCHWAB_APP_KEY')
SCHWAB_APP_SECRET = os.getenv('SCHWAB_APP_SECRET')
SCHWAB_CALLBACK_URL = os.getenv('SCHWAB_CALLBACK_URL')

# Token management
TOKENS_FILE = 'tokens.json'

# Force authentication at startup
print("\n" + "="*80)
print("SCHWAB API AUTHENTICATION")
print("="*80)
print("You need to authenticate with Schwab API to use this dashboard.")
print("Please follow the prompts below to complete authentication.")
print("="*80 + "\n")

# Import the Client class after displaying the authentication message
from schwabdev.client import Client

# Initialize the Schwab client - authentication is handled automatically by the library
client = Client(
    app_key=SCHWAB_APP_KEY,
    app_secret=SCHWAB_APP_SECRET,
    callback_url=SCHWAB_CALLBACK_URL,
    tokens_file=TOKENS_FILE
)

# Force token refresh to trigger authentication
if not os.path.exists(TOKENS_FILE) or os.path.getsize(TOKENS_FILE) == 0:
    print("No existing tokens found. Starting authentication process...")
    # Access tokens property to trigger authentication
    if hasattr(client, 'tokens'):
        client.tokens.update_refresh_token()

class OptionsDataRetriever:
    """
    Class to retrieve options data from Schwab API
    """
    def __init__(self, client):
        self.client = client
    
    def get_option_chain(self, symbol):
        """
        Get the option chain for a symbol
        
        Args:
            symbol (str): The stock symbol to get options for
            
        Returns:
            dict: Option chain data
        """
        try:
            # Get the current price of the underlying
            quote_response = self.client.quote(symbol)
            if hasattr(quote_response, 'json'):
                quote_data = quote_response.json()
                current_price = quote_data.get('lastPrice', 0)
            else:
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
            print(f"Error retrieving option chain: {str(e)}")
            return {
                "symbol": symbol,
                "underlying_price": 0,
                "expirations": [],
                "options": []
            }
    
    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1):
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): The stock symbol
            period_type (str): Type of period - 'day', 'month', 'year', 'ytd'
            period (int): Number of periods
            frequency_type (str): Type of frequency - 'minute', 'daily', 'weekly', 'monthly'
            frequency (int): Frequency value
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            # Get historical data
            history_response = self.client.price_history(
                symbol=symbol,
                periodType=period_type,
                period=period,
                frequencyType=frequency_type,
                frequency=frequency,
                needExtendedHoursData=True
            )
            
            if hasattr(history_response, 'json'):
                history_data = history_response.json()
            else:
                history_data = {}
            
            # Process the historical data
            candles = history_data.get('candles', [])
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            return pd.DataFrame()

# Initialize the options data retriever
options_data_retriever = OptionsDataRetriever(client)

# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Add debugging callback for timeline selector
add_debug_callback_to_app(app)

# Create the app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Schwab Options Dashboard", className="display-4"),
        html.P("Real-time options data and analysis powered by Schwab API", className="lead"),
        html.Hr()
    ], className="container mt-4"),
    
    # Main content
    html.Div([
        # Tabs
        dcc.Tabs(id="tabs", value="recommendations-tab", children=[
            dcc.Tab(label="Recommendations", value="recommendations-tab"),
            dcc.Tab(label="Technical Indicators", value="indicators-tab"),
            dcc.Tab(label="Greeks Analysis", value="greeks-tab"),
            dcc.Tab(label="Historical Data", value="historical-tab"),
            dcc.Tab(label="Real-Time Data", value="real-time-tab")
        ]),
        
        # Tab content
        html.Div(id="tab-content", className="p-4")
    ], className="container"),
    
    # Store components for sharing data between callbacks
    dcc.Store(id="options-data-store"),
    dcc.Store(id="historical-data-store"),
    dcc.Store(id="selected-options-store"),
    dcc.Store(id="streaming-data-store"),
    
    # Interval component for periodic updates
    dcc.Interval(
        id="interval-component",
        interval=60 * 1000,  # Update every 60 seconds
        n_intervals=0
    )
])

# Callback to render tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab):
    """
    Render the content for the selected tab
    
    Args:
        tab (str): Selected tab value
        
    Returns:
        html.Div: Tab content
    """
    if tab == "recommendations-tab":
        return create_recommendations_tab()
    elif tab == "indicators-tab":
        # Add the debug div to the indicators tab
        indicators_tab = create_indicators_tab()
        debug_div = add_debug_div_to_layout()
        
        # Combine the indicators tab with the debug div
        if isinstance(indicators_tab, html.Div) and hasattr(indicators_tab, 'children'):
            if isinstance(indicators_tab.children, list):
                indicators_tab.children.append(debug_div)
            else:
                indicators_tab.children = [indicators_tab.children, debug_div]
        
        return indicators_tab
    elif tab == "greeks-tab":
        return html.Div([
            html.H3("Options Greeks Analysis"),
            html.Div([
                html.Label("Symbol:"),
                dbc.InputGroup([
                    dbc.Input(id="greeks-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
                    dbc.Button("Analyze", id="greeks-analyze-button", color="primary")
                ], className="mb-3")
            ]),
            html.Div(id="greeks-content")
        ])
    elif tab == "historical-tab":
        return html.Div([
            html.H3("Historical Options Data"),
            html.Div([
                html.Label("Symbol:"),
                dbc.InputGroup([
                    dbc.Input(id="historical-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
                    dbc.Button("Fetch Data", id="historical-fetch-button", color="primary")
                ], className="mb-3")
            ]),
            html.Div(id="historical-content")
        ])
    elif tab == "real-time-tab":
        return html.Div([
            html.H3("Real-Time Options Data"),
            html.Div([
                html.Label("Symbol:"),
                dbc.InputGroup([
                    dbc.Input(id="real-time-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
                    dbc.Button("Start Streaming", id="real-time-start-button", color="primary"),
                    dbc.Button("Stop Streaming", id="real-time-stop-button", color="danger", className="ml-2")
                ], className="mb-3")
            ]),
            html.Div(id="real-time-content")
        ])
    else:
        return html.Div([
            html.H3("Tab content not implemented")
        ])

# Register callbacks
register_indicators_callbacks(app)
register_greeks_callbacks(app)
register_historical_callbacks(app)
register_real_time_callbacks(app)
register_recommendations_callbacks(app)

# Run the app
if __name__ == "__main__":
    # Open browser automatically
    webbrowser.open_new("http://127.0.0.1:8050/")
    
    # Run the app
    app.run_server(debug=True, use_reloader=False)
