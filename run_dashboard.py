"""
Full-featured Schwab Options Dashboard application with simplified import structure
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
        
        # Load tokens if they exist
        if os.path.exists(TOKENS_FILE):
            try:
                with open(TOKENS_FILE, 'r') as f:
                    self.tokens = json.load(f)
            except Exception as e:
                print(f"Error loading tokens: {str(e)}")
    
    def get_auth_status(self):
        """Check if we have valid authentication tokens"""
        return self.tokens is not None
    
    def authenticate(self, callback_url=None):
        """
        Authenticate with Schwab API
        
        Args:
            callback_url (str): Callback URL from OAuth flow
            
        Returns:
            bool: Whether authentication was successful
        """
        # This is a simplified placeholder for the authentication process
        # In a real implementation, this would handle the OAuth flow
        
        # For demo purposes, we'll just create a dummy token
        self.tokens = {
            "access_token": "dummy_access_token",
            "refresh_token": "dummy_refresh_token",
            "expires_in": 3600
        }
        
        # Save tokens
        with open(TOKENS_FILE, 'w') as f:
            json.dump(self.tokens, f)
        
        return True

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
        # This is a simplified placeholder that returns dummy data
        # In a real implementation, this would make API calls to Schwab
        
        # Generate some dummy option chain data
        expiration_dates = [
            (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
        ]
        
        current_price = 150.0  # Dummy current price
        
        option_chain = {
            "symbol": symbol,
            "underlying_price": current_price,
            "expirations": expiration_dates,
            "options": []
        }
        
        # Generate some dummy options for each expiration
        for exp in expiration_dates:
            for strike in range(int(current_price * 0.8), int(current_price * 1.2), 5):
                # Call option
                option_chain["options"].append({
                    "option_type": "CALL",
                    "symbol": f"{symbol}_{exp}_C_{strike}",
                    "strike": strike,
                    "expiration": exp,
                    "bid": round(max(0, current_price - strike + 5 + (0.1 * (current_price - strike))), 2),
                    "ask": round(max(0, current_price - strike + 5 + (0.1 * (current_price - strike))) + 0.5, 2),
                    "last": round(max(0, current_price - strike + 5 + (0.1 * (current_price - strike))) + 0.25, 2),
                    "volume": int(1000 * (1 - abs(current_price - strike) / current_price)),
                    "open_interest": int(5000 * (1 - abs(current_price - strike) / current_price)),
                    "delta": round(max(0, min(1, 0.5 + (current_price - strike) / 20)), 2),
                    "gamma": round(max(0, 0.05 - abs(current_price - strike) / 400), 4),
                    "theta": round(-0.05 - abs(current_price - strike) / 200, 4),
                    "vega": round(0.1 - abs(current_price - strike) / 300, 4),
                    "implied_volatility": round(0.3 + abs(current_price - strike) / 200, 2)
                })
                
                # Put option
                option_chain["options"].append({
                    "option_type": "PUT",
                    "symbol": f"{symbol}_{exp}_P_{strike}",
                    "strike": strike,
                    "expiration": exp,
                    "bid": round(max(0, strike - current_price + 5 + (0.1 * (strike - current_price))), 2),
                    "ask": round(max(0, strike - current_price + 5 + (0.1 * (strike - current_price))) + 0.5, 2),
                    "last": round(max(0, strike - current_price + 5 + (0.1 * (strike - current_price))) + 0.25, 2),
                    "volume": int(1000 * (1 - abs(current_price - strike) / current_price)),
                    "open_interest": int(5000 * (1 - abs(current_price - strike) / current_price)),
                    "delta": round(min(0, max(-1, -0.5 + (current_price - strike) / 20)), 2),
                    "gamma": round(max(0, 0.05 - abs(current_price - strike) / 400), 4),
                    "theta": round(-0.05 - abs(current_price - strike) / 200, 4),
                    "vega": round(0.1 - abs(current_price - strike) / 300, 4),
                    "implied_volatility": round(0.3 + abs(current_price - strike) / 200, 2)
                })
        
        return option_chain
    
    def get_historical_data(self, symbol, period="1M"):
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): The stock symbol
            period (str): Time period - '1D', '1W', '1M', '3M', '1Y'
            
        Returns:
            pd.DataFrame: Historical price data
        """
        # This is a simplified placeholder that returns dummy data
        # In a real implementation, this would make API calls to Schwab
        
        # Determine number of days based on period
        days = {
            "1D": 1,
            "1W": 7,
            "1M": 30,
            "3M": 90,
            "1Y": 365
        }.get(period, 30)
        
        # Generate dummy historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate dates
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only include weekdays
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Generate price data
        base_price = 150.0
        data = []
        
        for i, date in enumerate(dates):
            # Add some randomness to the price
            open_price = base_price + (i * 0.1) + ((-1) ** i) * (i % 5)
            close_price = open_price + ((-1) ** i) * (i % 3) * 0.5
            high_price = max(open_price, close_price) + (i % 3)
            low_price = min(open_price, close_price) - (i % 2)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": int(1000000 + (i % 10) * 100000)
            })
        
        return pd.DataFrame(data)

# Initialize authentication and data retriever
auth = SchwabAuth()
options_data = OptionsDataRetriever(auth)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# App layout
app.layout = html.Div([
    html.H1("Schwab Options Dashboard"),
    
    # Authentication status
    html.Div([
        html.Div(id="auth-status"),
        html.Button("Authenticate", id="auth-button", n_clicks=0, style={"display": "none"}),
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
     Output("auth-button", "style")],
    [Input("auth-button", "n_clicks")]
)
def check_auth_status(n_clicks):
    if auth.get_auth_status():
        return "Authentication Status: Authenticated", {"display": "none"}
    elif n_clicks > 0:
        # Attempt authentication
        if auth.authenticate():
            return "Authentication Status: Authenticated", {"display": "none"}
        else:
            return "Authentication Status: Failed", {"display": "block"}
    else:
        return "Authentication Status: Not Authenticated", {"display": "block"}

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
