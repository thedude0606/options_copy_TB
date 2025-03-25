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
            # Return None or empty data structure
            return {
                "symbol": symbol,
                "underlying_price": 0,
                "expirations": [],
                "options": []
            }
    
    def get_historical_data(self, symbol, period="1M"):
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): The stock symbol
            period (str): Time period - '1D', '1W', '1M', '3M', '1Y'
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            # Map period to frequency type and frequency
            period_mapping = {
                "1D": {"frequencyType": "minute", "frequency": 5},
                "1W": {"frequencyType": "minute", "frequency": 30},
                "1M": {"frequencyType": "daily", "frequency": 1},
                "3M": {"frequencyType": "daily", "frequency": 1},
                "1Y": {"frequencyType": "daily", "frequency": 1}
            }
            
            # Determine end date (now) and start date based on period
            end_date = datetime.now()
            
            days_mapping = {
                "1D": 1,
                "1W": 7,
                "1M": 30,
                "3M": 90,
                "1Y": 365
            }
            
            start_date = end_date - timedelta(days=days_mapping.get(period, 30))
            
            # Format dates for API
            start_ms = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            print(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            print(f"Period: {period}, FrequencyType: {period_mapping[period]['frequencyType']}, Frequency: {period_mapping[period]['frequency']}")
            
            # Get price history
            price_history_response = self.client.price_history(
                symbol=symbol,
                periodType="day" if period in ["1D", "1W"] else "month" if period in ["1M", "3M"] else "year",
                period=1 if period == "1D" else 5 if period == "1W" else 1 if period == "1M" else 3 if period == "3M" else 1,
                frequencyType=period_mapping[period]["frequencyType"],
                frequency=period_mapping[period]["frequency"],
                startDate=start_ms,
                endDate=end_ms
            )
            
            if hasattr(price_history_response, 'json'):
                price_history_data = price_history_response.json()
            else:
                price_history_data = {}
            
            # Process the price history data
            candles = price_history_data.get('candles', [])
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            if not df.empty:
                # Convert datetime
                df['date'] = pd.to_datetime(df['datetime'], unit='ms')
                
                # Rename columns
                df = df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
                
                # Select and order columns
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                
                # Sort by date
                df = df.sort_values('date')
                
                return df.to_dict('records')
            else:
                return []
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            return []
    
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
            quote_response = self.client.quote(symbol)
            
            if hasattr(quote_response, 'json'):
                quote_data = quote_response.json()
            else:
                quote_data = {}
            
            return quote_data
        except Exception as e:
            print(f"Error retrieving quote: {str(e)}")
            return {}

# Initialize the options data retriever
options_data = OptionsDataRetriever(client)

# Initialize the streaming data manager
streaming_manager = StreamingDataManager(interactive_auth=True)

# Initialize the stream data handler
stream_handler = StreamDataHandler()

# Initialize data collector and recommendation engine
data_collector = DataCollector(interactive_auth=False)
recommendation_engine = RecommendationEngine(data_collector)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Register callbacks
register_real_time_callbacks(app)
register_indicators_callbacks(app)
register_greeks_callbacks(app)
register_historical_callbacks(app)
register_recommendations_callbacks(app, recommendation_engine)

# Define callback to handle stream data
@app.callback(
    Output("rt-stream-data", "data"),
    [Input("rt-update-interval", "n_intervals")],
    [State("rt-symbols-store", "data"),
     State("rt-connection-store", "data"),
     State("rt-stream-data", "data")]
)
def update_stream_data(n_intervals, symbols, connection_data, current_data):
    if not symbols or not connection_data or not connection_data.get("active", False):
        return {}
    
    # Get data from stream handler
    data_store = stream_handler.get_data_store()
    
    # If no current data, initialize with data store
    if not current_data:
        return data_store
    
    # Otherwise, update with new data
    return data_store

# Callback to start/stop streaming
@app.callback(
    Output("rt-connection-status", "children", allow_duplicate=True),
    [Input("rt-start-stream-button", "n_clicks"),
     Input("rt-stop-stream-button", "n_clicks")],
    [State("rt-symbols-store", "data"),
     State("rt-data-type", "value"),
     State("rt-connection-store", "data")],
    prevent_initial_call=True
)
def manage_streaming(start_clicks, stop_clicks, symbols, data_type, connection_data):
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "rt-start-stream-button" and start_clicks > 0:
        # Start streaming
        if not symbols:
            return html.Div([
                html.Span("Error", style={"color": "red"}),
                html.Span(" - No symbols selected", style={"font-style": "italic", "margin-left": "5px"})
            ])
        
        try:
            # Define callback for stream data
            def stream_callback(message):
                processed_data = stream_handler.process_message(message)
                return processed_data
            
            # Start stream
            streaming_manager.start_stream(stream_callback)
            
            # Subscribe to appropriate data
            if data_type == "quotes":
                streaming_manager.subscribe_level1_quotes(symbols)
            else:
                streaming_manager.subscribe_option_quotes(symbols)
            
            return html.Div([
                html.Span("Connected", style={"color": "green"}),
                html.Span(f" - Streaming {data_type} for {', '.join(symbols)}", style={"font-style": "italic", "margin-left": "5px"})
            ])
        except Exception as e:
            return html.Div([
                html.Span("Error", style={"color": "red"}),
                html.Span(f" - {str(e)}", style={"font-style": "italic", "margin-left": "5px"})
            ])
    
    elif button_id == "rt-stop-stream-button" and stop_clicks > 0:
        # Stop streaming
        try:
            streaming_manager.stop_stream()
            stream_handler.clear_data_store()
            
            return html.Div([
                html.Span("Disconnected", style={"color": "red"}),
                html.Span(" - Stream stopped", style={"font-style": "italic", "margin-left": "5px"})
            ])
        except Exception as e:
            return html.Div([
                html.Span("Error", style={"color": "red"}),
                html.Span(f" - {str(e)}", style={"font-style": "italic", "margin-left": "5px"})
            ])
    
    return dash.no_update

# App layout
app.layout = html.Div([
    html.H1("Schwab Options Dashboard"),
    
    # Authentication status
    html.Div(id="auth-status", children=[
        html.Div("Checking authentication status...", id="auth-message"),
        html.Div(id="auth-details", style={"display": "none"})
    ], style={"border": "1px solid blue", "padding": "10px", "margin": "10px 0", "background-color": "#f0f8ff"}),
    
    # Symbol input and submit button
    html.Div([
        html.Label("Enter Symbol:"),
        dcc.Input(id="symbol-input", type="text", value="AAPL", placeholder="Enter stock symbol"),
        html.Button("Submit", id="submit-button", n_clicks=0),
    ], style={"margin": "20px"}),
    
    # Tabs for different views
    dcc.Tabs([
        # Options Chain Tab
        dcc.Tab(label="Options Chain", children=[
            html.Div([
                # Expiration date dropdown
                html.Label("Expiration Date:"),
                dcc.Dropdown(id="expiration-dropdown", placeholder="Select expiration date"),
                
                # Option type radio buttons
                html.Label("Option Type:"),
                dcc.RadioItems(
                    id="option-type",
                    options=[
                        {"label": "Calls", "value": "CALL"},
                        {"label": "Puts", "value": "PUT"},
                        {"label": "Both", "value": "ALL"}
                    ],
                    value="ALL",
                    inline=True
                ),
                
                # Options chain table
                html.Div(id="options-chain-container")
            ])
        ]),
        
        # Greeks Tab
        dcc.Tab(label="Greeks", children=[
            html.Div([
                # Expiration date dropdown for Greeks
                html.Label("Expiration Date:"),
                dcc.Dropdown(id="greeks-expiration-dropdown", placeholder="Select expiration date"),
                
                # Greeks visualization
                html.Div(id="greeks-container")
            ])
        ]),
        
        # Historical Data Tab
        dcc.Tab(label="Historical Data", children=[
            html.Div([
                # Time period selection
                html.Label("Time Period:"),
                dcc.Dropdown(
                    id="time-period",
                    options=[
                        {"label": "1 Day", "value": "1D"},
                        {"label": "1 Week", "value": "1W"},
                        {"label": "1 Month", "value": "1M"},
                        {"label": "3 Months", "value": "3M"},
                        {"label": "1 Year", "value": "1Y"}
                    ],
                    value="1M"
                ),
                
                # Candle chart
                dcc.Graph(id="historical-chart")
            ])
        ]),
        
        # Technical Indicators Tab
        dcc.Tab(label="Technical Indicators", children=[
            create_indicators_tab()
        ]),
        
        # Recommendations Tab
        dcc.Tab(label="Recommendations", children=[
            create_recommendations_tab()
        ]),
        
        # Real-Time Data Tab
        dcc.Tab(label="Real-Time Data", children=[
            html.Div([
                # Controls section
                html.Div([
                    html.H3("Real-Time Data Controls"),
                    
                    # Symbol selection
                    html.Div([
                        html.Label("Symbol:"),
                        dcc.Input(
                            id="rt-symbol-input",
                            type="text",
                            value="",
                            placeholder="Enter symbol (e.g., AAPL)",
                            style={"width": "150px", "margin-right": "10px"}
                        ),
                        html.Button(
                            "Add Symbol",
                            id="rt-add-symbol-button",
                            n_clicks=0,
                            style={"margin-right": "10px"}
                        ),
                        
                        # Display active symbols
                        html.Div([
                            html.Label("Active Symbols:"),
                            html.Div(id="rt-active-symbols", style={"margin-top": "5px"})
                        ], style={"margin-top": "10px"})
                    ], style={"margin-bottom": "20px"}),
                    
                    # Connection controls
                    html.Div([
                        html.Button(
                            "Start Stream",
                            id="rt-start-stream-button",
                            n_clicks=0,
                            style={"margin-right": "10px"}
                        ),
                        html.Button(
                            "Stop Stream",
                            id="rt-stop-stream-button",
                            n_clicks=0,
                            style={"margin-right": "10px"}
                        ),
                        html.Div(id="rt-connection-status", style={"margin-top": "5px"})
                    ], style={"margin-bottom": "20px"}),
                    
                    # Data type selection
                    html.Div([
                        html.Label("Data Type:"),
                        dcc.RadioItems(
                            id="rt-data-type",
                            options=[
                                {"label": "Quotes", "value": "quotes"},
                                {"label": "Options", "value": "options"}
                            ],
                            value="quotes",
                            inline=True
                        )
                    ], style={"margin-bottom": "20px"})
                ], style={"padding": "15px", "background-color": "#f8f9fa", "border-radius": "5px"}),
                
                # Data display section
                html.Div([
                    html.H3("Real-Time Data"),
                    
                    # Tabs for different data views
                    dcc.Tabs([
                        # Price chart tab
                        dcc.Tab(label="Price Chart", children=[
                            dcc.Graph(id="rt-price-chart")
                        ]),
                        
                        # Data table tab
                        dcc.Tab(label="Data Table", children=[
                            html.Div(id="rt-data-table")
                        ]),
                        
                        # Time & Sales tab
                        dcc.Tab(label="Time & Sales", children=[
                            html.Div(id="rt-time-sales")
                        ])
                    ])
                ], style={"margin-top": "20px"}),
                
                # Hidden divs for storing data
                dcc.Store(id="rt-stream-data"),
                dcc.Store(id="rt-symbols-store"),
                dcc.Store(id="rt-connection-store"),
                
                # Interval component for updating charts
                dcc.Interval(
                    id="rt-update-interval",
                    interval=1000,  # 1 second
                    n_intervals=0,
                    disabled=True
                )
            ])
        ])
    ]),
    
    # Store component for holding data
    dcc.Store(id="options-data"),
    dcc.Store(id="historical-data"),
    dcc.Store(id="quote-data")
])

# Callback to check authentication status
@app.callback(
    [Output("auth-message", "children"),
     Output("auth-details", "children"),
     Output("auth-details", "style")],
    [Input("auth-status", "id")]
)
def update_auth_status(auth_id):
    if hasattr(client, 'tokens') and hasattr(client.tokens, 'access_token') and client.tokens.access_token:
        # Authentication successful
        message = html.Div([
            html.Span("✓ ", style={"color": "green", "font-weight": "bold"}),
            "Authentication successful"
        ])
        
        # Get account details
        try:
            accounts_response = client.account_details_all()
            if hasattr(accounts_response, 'json'):
                accounts_data = accounts_response.json()
                
                # Extract account details
                accounts = []
                for account in accounts_data:
                    account_id = account.get('accountNumber', 'Unknown')
                    account_type = account.get('type', 'Unknown')
                    accounts.append(f"Account: {account_id} (Type: {account_type})")
                
                details = html.Div([
                    html.H4("Account Information"),
                    html.Ul([html.Li(account) for account in accounts])
                ])
                
                return message, details, {"display": "block", "margin-top": "10px"}
            else:
                return message, "No account details available", {"display": "block", "margin-top": "10px"}
        except Exception as e:
            return message, f"Error retrieving account details: {str(e)}", {"display": "block", "margin-top": "10px"}
    else:
        # Authentication failed
        message = html.Div([
            html.Span("✗ ", style={"color": "red", "font-weight": "bold"}),
            "Authentication failed or not completed"
        ])
        return message, "Please check your credentials and try again", {"display": "block", "margin-top": "10px"}

# Callback to fetch data when symbol is submitted
@app.callback(
    [Output("options-data", "data"),
     Output("historical-data", "data"),
     Output("quote-data", "data"),
     Output("expiration-dropdown", "options"),
     Output("greeks-expiration-dropdown", "options")],
    [Input("submit-button", "n_clicks")],
    [State("symbol-input", "value"),
     State("time-period", "value")]
)
def fetch_data(n_clicks, symbol, time_period):
    if n_clicks == 0:
        # Default data for initial load
        return None, None, None, [], []
    
    if not symbol:
        return None, None, None, [], []
    
    # Get option chain
    option_chain = options_data.get_option_chain(symbol)
    
    # Get historical data
    historical_data = options_data.get_historical_data(symbol, period=time_period)
    
    # Get current quote
    quote = options_data.get_quote(symbol)
    
    # Extract expiration dates for dropdown
    expiration_dates = []
    if option_chain and "expirations" in option_chain:
        expiration_dates = [{"label": exp, "value": exp} for exp in option_chain["expirations"]]
    
    return option_chain, historical_data, quote, expiration_dates, expiration_dates

# Callback to update options chain display
@app.callback(
    Output("options-chain-container", "children"),
    [Input("options-data", "data"),
     Input("expiration-dropdown", "value"),
     Input("option-type", "value")]
)
def update_options_chain(options_data, expiration, option_type):
    if not options_data or not expiration:
        return html.Div("No data available. Please enter a symbol and select an expiration date.")
    
    # Filter options by expiration and type
    filtered_options = []
    for option in options_data.get("options", []):
        if option["expiration"] == expiration:
            if option_type == "ALL" or option["option_type"] == option_type:
                filtered_options.append(option)
    
    if not filtered_options:
        return html.Div("No options found for the selected criteria.")
    
    # Sort by strike price
    filtered_options.sort(key=lambda x: x["strike"])
    
    # Create table
    table_header = [
        html.Thead(html.Tr([
            html.Th("Type"),
            html.Th("Symbol"),
            html.Th("Strike"),
            html.Th("Bid"),
            html.Th("Ask"),
            html.Th("Last"),
            html.Th("Volume"),
            html.Th("Open Int"),
            html.Th("Delta"),
            html.Th("Gamma"),
            html.Th("Theta"),
            html.Th("Vega"),
            html.Th("IV")
        ]))
    ]
    
    rows = []
    for option in filtered_options:
        row = html.Tr([
            html.Td(option["option_type"]),
            html.Td(option["symbol"]),
            html.Td(f"${option['strike']:.2f}"),
            html.Td(f"${option['bid']:.2f}"),
            html.Td(f"${option['ask']:.2f}"),
            html.Td(f"${option['last']:.2f}"),
            html.Td(f"{option['volume']:,}"),
            html.Td(f"{option['open_interest']:,}"),
            html.Td(f"{option['delta']:.3f}"),
            html.Td(f"{option['gamma']:.3f}"),
            html.Td(f"{option['theta']:.3f}"),
            html.Td(f"{option['vega']:.3f}"),
            html.Td(f"{option['implied_volatility']:.2%}")
        ])
        rows.append(row)
    
    table_body = [html.Tbody(rows)]
    
    table = dbc.Table(
        table_header + table_body,
        bordered=True,
        striped=True,
        hover=True,
        responsive=True
    )
    
    return html.Div([
        html.H3(f"Options Chain for {options_data.get('symbol', '')} - {expiration}"),
        html.P(f"Underlying Price: ${options_data.get('underlying_price', 0):.2f}"),
        table
    ])

# Callback to update Greeks visualization
@app.callback(
    Output("greeks-container", "children"),
    [Input("options-data", "data"),
     Input("greeks-expiration-dropdown", "value")]
)
def update_greeks(options_data, expiration):
    if not options_data or not expiration:
        return html.Div("No data available. Please enter a symbol and select an expiration date.")
    
    # Filter options by expiration
    filtered_options = []
    for option in options_data.get("options", []):
        if option["expiration"] == expiration:
            filtered_options.append(option)
    
    if not filtered_options:
        return html.Div("No options found for the selected expiration date.")
    
    # Separate calls and puts
    calls = [opt for opt in filtered_options if opt["option_type"] == "CALL"]
    puts = [opt for opt in filtered_options if opt["option_type"] == "PUT"]
    
    # Sort by strike price
    calls.sort(key=lambda x: x["strike"])
    puts.sort(key=lambda x: x["strike"])
    
    # Create graphs
    graphs = []
    
    # Delta graph
    delta_fig = go.Figure()
    delta_fig.add_trace(go.Scatter(
        x=[option["strike"] for option in calls],
        y=[option["delta"] for option in calls],
        mode='lines+markers',
        name='Call Delta'
    ))
    delta_fig.add_trace(go.Scatter(
        x=[option["strike"] for option in puts],
        y=[option["delta"] for option in puts],
        mode='lines+markers',
        name='Put Delta'
    ))
    delta_fig.update_layout(
        title="Delta vs Strike Price",
        xaxis_title="Strike Price",
        yaxis_title="Delta",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    graphs.append(dcc.Graph(figure=delta_fig))
    
    # Gamma graph
    gamma_fig = go.Figure()
    gamma_fig.add_trace(go.Scatter(
        x=[option["strike"] for option in calls],
        y=[option["gamma"] for option in calls],
        mode='lines+markers',
        name='Call Gamma'
    ))
    gamma_fig.add_trace(go.Scatter(
        x=[option["strike"] for option in puts],
        y=[option["gamma"] for option in puts],
        mode='lines+markers',
        name='Put Gamma'
    ))
    gamma_fig.update_layout(
        title="Gamma vs Strike Price",
        xaxis_title="Strike Price",
        yaxis_title="Gamma",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    graphs.append(dcc.Graph(figure=gamma_fig))
    
    # Theta graph
    theta_fig = go.Figure()
    theta_fig.add_trace(go.Scatter(
        x=[option["strike"] for option in calls],
        y=[option["theta"] for option in calls],
        mode='lines+markers',
        name='Call Theta'
    ))
    theta_fig.add_trace(go.Scatter(
        x=[option["strike"] for option in puts],
        y=[option["theta"] for option in puts],
        mode='lines+markers',
        name='Put Theta'
    ))
    theta_fig.update_layout(
        title="Theta vs Strike Price",
        xaxis_title="Strike Price",
        yaxis_title="Theta",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    graphs.append(dcc.Graph(figure=theta_fig))
    
    # Vega graph
    vega_fig = go.Figure()
    vega_fig.add_trace(go.Scatter(
        x=[option["strike"] for option in calls],
        y=[option["vega"] for option in calls],
        mode='lines+markers',
        name='Call Vega'
    ))
    vega_fig.add_trace(go.Scatter(
        x=[option["strike"] for option in puts],
        y=[option["vega"] for option in puts],
        mode='lines+markers',
        name='Put Vega'
    ))
    vega_fig.update_layout(
        title="Vega vs Strike Price",
        xaxis_title="Strike Price",
        yaxis_title="Vega",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    graphs.append(dcc.Graph(figure=vega_fig))
    
    return html.Div(graphs)

# Callback to update historical chart
@app.callback(
    Output("historical-chart", "figure"),
    [Input("historical-data", "data"),
     Input("time-period", "value")]
)
def update_historical_chart(historical_data, time_period):
    print(f"update_historical_chart called with time_period: {time_period}")
    print(f"historical_data type: {type(historical_data)}")
    print(f"historical_data length: {len(historical_data) if historical_data else 0}")
    
    if not historical_data:
        print("No historical data available")
        return go.Figure()
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    print(f"DataFrame created with shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    if df.empty:
        print("DataFrame is empty")
        return go.Figure()
        
    print(f"DataFrame head: \n{df.head()}")
    
    # Convert date strings to datetime objects if they're not already
    if "date" in df.columns:
        # Check if dates are in milliseconds (epoch time) and convert if needed
        if isinstance(df["date"].iloc[0], (int, float)) or (isinstance(df["date"].iloc[0], str) and df["date"].iloc[0].isdigit()):
            df["date"] = pd.to_datetime(df["date"].astype(float), unit='ms')
        else:
            # Try to parse as datetime if it's a string in a different format
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
    
    # Create candlestick chart
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price"
    ))
    
    # Add a line chart for daily close prices
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["close"],
        mode='lines',
        name='Daily Close',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=f"Historical Price Data - {time_period}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    
    print("Historical chart figure created successfully")
    return fig

# Run the app
if __name__ == "__main__":
    # Force authentication check before starting the server
    print("\nChecking authentication status...")
    if hasattr(client, 'tokens') and hasattr(client.tokens, 'access_token') and client.tokens.access_token:
        print("Authentication successful! Starting dashboard server...\n")
    else:
        print("Authentication required. Please follow the prompts above.\n")
    
    app.run_server(debug=True, host="0.0.0.0", port=8050)
