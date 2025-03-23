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
from schwabdev.client import Client  # Correct import for the Schwab client

# Load environment variables
load_dotenv()

# Authentication and API credentials
SCHWAB_APP_KEY = os.getenv('SCHWAB_APP_KEY')
SCHWAB_APP_SECRET = os.getenv('SCHWAB_APP_SECRET')
SCHWAB_CALLBACK_URL = os.getenv('SCHWAB_CALLBACK_URL')

# Token management
TOKENS_FILE = 'tokens.json'

class SchwabAuth:
    """
    Class to handle Schwab API authentication
    """
    def __init__(self):
        self.app_key = SCHWAB_APP_KEY
        self.app_secret = SCHWAB_APP_SECRET
        self.callback_url = SCHWAB_CALLBACK_URL
        self.tokens = None
        self.client = None
        
        # Load tokens if they exist
        if os.path.exists(TOKENS_FILE):
            try:
                with open(TOKENS_FILE, 'r') as f:
                    self.tokens = json.load(f)
                    self.initialize_client()
            except Exception as e:
                print(f"Error loading tokens: {str(e)}")
    
    def initialize_client(self):
        """Initialize the Schwab client with existing tokens"""
        if self.tokens:
            try:
                # Use the correct Client class from schwabdev.client
                self.client = Client(
                    app_key=self.app_key,
                    app_secret=self.app_secret,
                    callback_url=self.callback_url,
                    tokens_file=TOKENS_FILE
                )
                return True
            except Exception as e:
                print(f"Error initializing client: {str(e)}")
                return False
        return False
    
    def get_auth_status(self):
        """Check if we have valid authentication tokens and client"""
        return self.tokens is not None and self.client is not None
    
    def authenticate(self, callback_url=None):
        """
        Authenticate with Schwab API
        
        Args:
            callback_url (str): Callback URL from OAuth flow
            
        Returns:
            bool: Whether authentication was successful
        """
        try:
            # Initialize client for authentication
            self.client = Client(
                app_key=self.app_key,
                app_secret=self.app_secret,
                callback_url=self.callback_url,
                tokens_file=TOKENS_FILE
            )
            
            if callback_url:
                # Complete authentication with callback URL
                self.client.get_access_token(callback_url)
                
                # Save tokens
                self.tokens = {
                    "access_token": self.client.access_token,
                    "refresh_token": self.client.refresh_token,
                    "expires_in": 3600  # Default expiration
                }
                
                with open(TOKENS_FILE, 'w') as f:
                    json.dump(self.tokens, f)
                
                return True
            else:
                # Start authentication flow
                auth_url = self.client.get_auth_url()
                print(f"Please visit this URL to authenticate: {auth_url}")
                webbrowser.open(auth_url)
                return False
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False

class OptionsDataRetriever:
    """
    Class to retrieve options data from Schwab API
    """
    def __init__(self, auth):
        self.auth = auth
    
    def get_option_chain(self, symbol):
        """
        Get the option chain for a symbol
        
        Args:
            symbol (str): The stock symbol to get options for
            
        Returns:
            dict: Option chain data
        """
        if not self.auth.get_auth_status():
            print("Not authenticated. Please authenticate first.")
            return None
        
        try:
            # Get the current price of the underlying
            quote = self.auth.client.get_quote(symbol)
            current_price = quote.get('lastPrice', 0)
            
            # Get option chain data
            option_chain_data = self.auth.client.get_option_chain(
                symbol=symbol,
                contract_type="ALL",
                strike_count=10,  # Get options around the current price
                include_quotes=True,
                strategy="SINGLE"
            )
            
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
        if not self.auth.get_auth_status():
            print("Not authenticated. Please authenticate first.")
            return pd.DataFrame()
        
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
            
            # Get price history
            price_history = self.auth.client.get_price_history(
                symbol=symbol,
                start_date=start_ms,
                end_date=end_ms,
                frequency_type=period_mapping[period]["frequencyType"],
                frequency=period_mapping[period]["frequency"]
            )
            
            # Process the price history data
            candles = price_history.get('candles', [])
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    "date": datetime.fromtimestamp(candle.get('datetime', 0) / 1000).strftime("%Y-%m-%d"),
                    "open": candle.get('open', 0),
                    "high": candle.get('high', 0),
                    "low": candle.get('low', 0),
                    "close": candle.get('close', 0),
                    "volume": candle.get('volume', 0)
                })
            
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            return pd.DataFrame()

# Initialize authentication and data retriever
auth = SchwabAuth()
options_data = OptionsDataRetriever(auth)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# App layout
app.layout = html.Div([
    html.H1("Schwab Options Dashboard"),
    
    # Authentication status and controls
    html.Div([
        html.Div(id="auth-status"),
        html.Button("Authenticate", id="auth-button", n_clicks=0),
        dcc.Input(id="callback-url", type="text", placeholder="Paste callback URL here", style={"width": "50%", "display": "none"}),
        html.Button("Submit Callback URL", id="callback-submit", n_clicks=0, style={"display": "none"}),
    ], style={"margin": "10px"}),
    
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
        ])
    ]),
    
    # Store component for holding data
    dcc.Store(id="options-data"),
    dcc.Store(id="historical-data"),
    dcc.Store(id="quote-data")
])

# Callback to check authentication status
@app.callback(
    [Output("auth-status", "children"),
     Output("auth-button", "style"),
     Output("callback-url", "style"),
     Output("callback-submit", "style")],
    [Input("auth-button", "n_clicks"),
     Input("callback-submit", "n_clicks")],
    [State("callback-url", "value")]
)
def check_auth_status(auth_clicks, callback_clicks, callback_url):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if auth.get_auth_status():
        return (
            "Authentication Status: Authenticated", 
            {"display": "none"}, 
            {"display": "none", "width": "50%"}, 
            {"display": "none"}
        )
    elif triggered_id == "auth-button" and auth_clicks > 0:
        # Start authentication flow
        auth.authenticate()
        return (
            "Authentication Status: Please complete authentication in browser and paste callback URL", 
            {"display": "none"}, 
            {"display": "block", "width": "50%"}, 
            {"display": "block"}
        )
    elif triggered_id == "callback-submit" and callback_clicks > 0 and callback_url:
        # Complete authentication with callback URL
        if auth.authenticate(callback_url):
            return (
                "Authentication Status: Authenticated", 
                {"display": "none"}, 
                {"display": "none", "width": "50%"}, 
                {"display": "none"}
            )
        else:
            return (
                "Authentication Status: Authentication Failed", 
                {"display": "block"}, 
                {"display": "block", "width": "50%"}, 
                {"display": "block"}
            )
    else:
        return (
            "Authentication Status: Not Authenticated", 
            {"display": "block"}, 
            {"display": "none", "width": "50%"}, 
            {"display": "none"}
        )

# Callback to fetch data when symbol is submitted
@app.callback(
    [Output("options-data", "data"),
     Output("historical-data", "data"),
     Output("expiration-dropdown", "options"),
     Output("greeks-expiration-dropdown", "options")],
    [Input("submit-button", "n_clicks")],
    [State("symbol-input", "value"),
     State("time-period", "value")]
)
def fetch_data(n_clicks, symbol, time_period):
    if n_clicks == 0:
        # Default data for initial load
        return None, None, [], []
    
    if not symbol:
        return None, None, [], []
    
    if not auth.get_auth_status():
        return None, None, [], []
    
    # Get option chain
    option_chain = options_data.get_option_chain(symbol)
    
    # Get historical data
    historical_data = options_data.get_historical_data(symbol, time_period)
    
    # Extract expiration dates for dropdown
    expiration_dates = []
    if option_chain and "expirations" in option_chain:
        expiration_dates = [{"label": exp, "value": exp} for exp in option_chain["expirations"]]
    
    return option_chain, historical_data.to_dict('records'), expiration_dates, expiration_dates

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
    
    # Filter options data by expiration and type
    filtered_options = []
    for option in options_data["options"]:
        if option["expiration"] == expiration:
            if option_type == "ALL" or option["option_type"] == option_type:
                filtered_options.append(option)
    
    # Sort by strike price
    filtered_options.sort(key=lambda x: x["strike"])
    
    # Create table to display options data
    columns = [
        {"name": "Type", "id": "option_type"},
        {"name": "Strike", "id": "strike"},
        {"name": "Bid", "id": "bid"},
        {"name": "Ask", "id": "ask"},
        {"name": "Last", "id": "last"},
        {"name": "Volume", "id": "volume"},
        {"name": "Open Int", "id": "open_interest"},
        {"name": "Delta", "id": "delta"},
        {"name": "Gamma", "id": "gamma"},
        {"name": "Theta", "id": "theta"},
        {"name": "Vega", "id": "vega"},
        {"name": "IV", "id": "implied_volatility"}
    ]
    
    return dash_table.DataTable(
        id='options-table',
        columns=columns,
        data=filtered_options,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'filter_query': '{option_type} = "CALL"'},
                'backgroundColor': 'rgba(0, 128, 0, 0.1)'
            },
            {
                'if': {'filter_query': '{option_type} = "PUT"'},
                'backgroundColor': 'rgba(255, 0, 0, 0.1)'
            }
        ]
    )

# Callback to update Greeks visualization
@app.callback(
    Output("greeks-container", "children"),
    [Input("options-data", "data"),
     Input("greeks-expiration-dropdown", "value")]
)
def update_greeks(options_data, expiration):
    if not options_data or not expiration:
        return html.Div("No data available. Please enter a symbol and select an expiration date.")
    
    # Filter options data by expiration
    filtered_options = [option for option in options_data["options"] if option["expiration"] == expiration]
    
    # Separate calls and puts
    calls = [option for option in filtered_options if option["option_type"] == "CALL"]
    puts = [option for option in filtered_options if option["option_type"] == "PUT"]
    
    # Sort by strike price
    calls.sort(key=lambda x: x["strike"])
    puts.sort(key=lambda x: x["strike"])
    
    # Create graphs for each Greek
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
    if not historical_data:
        return go.Figure()
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
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
    
    fig.update_layout(
        title=f"Historical Price Data - {time_period}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
