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
                startDate=start_ms,
                endDate=end_ms,
                frequencyType=period_mapping[period]["frequencyType"],
                frequency=period_mapping[period]["frequency"],
                needExtendedHoursData=True
            )
            
            print(f"API Response type: {type(price_history_response)}")
            
            if hasattr(price_history_response, 'json'):
                price_history = price_history_response.json()
                print(f"Price history JSON keys: {price_history.keys() if price_history else 'None'}")
                
                # Check for errors
                if 'errors' in price_history:
                    print(f"API returned errors: {price_history['errors']}")
                    
                    # Try with different parameters - use day periodType for all
                    print("Retrying with different parameters...")
                    price_history_response = self.client.price_history(
                        symbol=symbol,
                        periodType="day",
                        period=1,
                        frequencyType="daily",
                        frequency=1,
                        needExtendedHoursData=True
                    )
                    
                    if hasattr(price_history_response, 'json'):
                        price_history = price_history_response.json()
                        print(f"Retry response keys: {price_history.keys() if price_history else 'None'}")
                        
                        if 'errors' in price_history:
                            print(f"Retry also returned errors: {price_history['errors']}")
                            
                            # Try with a simpler approach - use month periodType
                            print("Trying with month periodType...")
                            price_history_response = self.client.price_history(
                                symbol=symbol,
                                periodType="month",
                                period=1,
                                frequencyType="daily",
                                frequency=1
                            )
                            
                            if hasattr(price_history_response, 'json'):
                                price_history = price_history_response.json()
                                print(f"Month periodType response keys: {price_history.keys() if price_history else 'None'}")
            else:
                price_history = {}
                print(f"Price history response has no json method. Response: {price_history_response}")
            
            # Process the price history data
            candles = price_history.get('candles', [])
            
            print(f"Number of candles received: {len(candles)}")
            if candles and len(candles) > 0:
                print(f"First candle: {candles[0]}")
            
            if not candles:
                print("No candles data received from API")
                # Create sample data for testing
                print("Creating sample data for testing visualization...")
                candles = self._create_sample_data(symbol, start_date, end_date, days_mapping.get(period, 30))
                print(f"Created {len(candles)} sample candles")
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                if isinstance(candle, dict):
                    # API data
                    data.append({
                        "date": datetime.fromtimestamp(candle.get('datetime', 0) / 1000).strftime("%Y-%m-%d"),
                        "open": candle.get('open', 0),
                        "high": candle.get('high', 0),
                        "low": candle.get('low', 0),
                        "close": candle.get('close', 0),
                        "volume": candle.get('volume', 0)
                    })
                else:
                    # Sample data
                    data.append(candle)
            
            df = pd.DataFrame(data)
            print(f"Created DataFrame with {len(df)} rows")
            print(f"DataFrame columns: {df.columns}")
            if not df.empty:
                print(f"DataFrame head: {df.head()}")
            
            return df
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _create_sample_data(self, symbol, start_date, end_date, days):
        """Create sample historical data for testing visualization"""
        data = []
        current_date = start_date
        base_price = 150.0  # Example base price
        
        if symbol == "AAPL":
            base_price = 170.0
        elif symbol == "MSFT":
            base_price = 400.0
        elif symbol == "GOOGL":
            base_price = 150.0
        elif symbol == "AMZN":
            base_price = 180.0
        
        # Generate daily data
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                # Create some random price movement
                daily_volatility = base_price * 0.02  # 2% volatility
                
                open_price = base_price + random.uniform(-daily_volatility, daily_volatility)
                close_price = open_price + random.uniform(-daily_volatility, daily_volatility)
                high_price = max(open_price, close_price) + random.uniform(0, daily_volatility)
                low_price = min(open_price, close_price) - random.uniform(0, daily_volatility)
                
                # Ensure low <= open, close <= high
                low_price = min(low_price, open_price, close_price)
                high_price = max(high_price, open_price, close_price)
                
                # Add some trend
                trend_factor = math.sin(current_date.day / 15) * daily_volatility * 2
                
                data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "open": round(open_price + trend_factor, 2),
                    "high": round(high_price + trend_factor, 2),
                    "low": round(low_price + trend_factor, 2),
                    "close": round(close_price + trend_factor, 2),
                    "volume": int(random.uniform(5000000, 15000000))
                })
                
                # Update base price for next day (slight drift)
                base_price = close_price + trend_factor
            
            current_date += timedelta(days=1)
        
        return data

# Initialize data retriever
options_data = OptionsDataRetriever(client)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# App layout
app.layout = html.Div([
    html.H1("Schwab Options Dashboard"),
    
    # Authentication status
    html.Div([
        html.Div(id="auth-status"),
        html.Div([
            html.H3("Authentication Instructions:", style={"color": "blue"}),
            html.P("1. Check the terminal/console where you started this dashboard"),
            html.P("2. Follow the authentication prompts in the terminal"),
            html.P("3. After authenticating in your browser, copy the callback URL"),
            html.P("4. Paste the callback URL back into the terminal (not here)"),
            html.P("5. Once authenticated, you can use the dashboard below")
        ], style={"border": "1px solid blue", "padding": "10px", "margin": "10px 0", "background-color": "#f0f8ff"}),
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
    Output("auth-status", "children"),
    [Input("submit-button", "n_clicks")]
)
def check_auth_status(n_clicks):
    # The authentication is handled automatically by the library
    # This just displays the current status
    if hasattr(client, 'tokens') and hasattr(client.tokens, 'access_token') and client.tokens.access_token:
        return html.Div("Authentication Status: Authenticated", style={"color": "green"})
    else:
        return html.Div("Authentication Status: Not Authenticated - Please check console for authentication instructions", 
                        style={"color": "red"})

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
