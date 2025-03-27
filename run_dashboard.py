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
            
            return df
        except Exception as e:
            print(f"Error retrieving historical data: {str(e)}")
            return pd.DataFrame()

# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Initialize data retriever
data_retriever = OptionsDataRetriever(client)

# Initialize data collector for recommendations
data_collector = DataCollector()

# Initialize recommendation engine
recommendation_engine = RecommendationEngine(data_collector)

# Initialize streaming data manager
streaming_manager = StreamingDataManager(client)

# Initialize stream data handler
stream_handler = StreamDataHandler(streaming_manager)

# App layout
app.layout = html.Div([
    # Header with title and global symbol search
    html.Div([
        html.H1("Options Recommendation Platform", className="app-title"),
        
        # Global symbol search
        html.Div([
            dcc.Input(id="global-symbol-input", type="text", value="AAPL", placeholder="Enter symbol"),
            html.Button("Search", id="global-search-button", className="search-button")
        ], className="symbol-search-container")
    ], className="app-header"),
    
    # Authentication status
    html.Div(id="auth-status", children=[
        html.Div("Checking authentication status...", id="auth-message"),
        html.Div(id="auth-details", style={"display": "none"})
    ], className="auth-status-container"),
    
    # Main content area
    html.Div([
        # Left sidebar
        html.Div([
            # Trading Timeframe section
            html.Div([
                html.H3("Trading Timeframe"),
                dcc.RadioItems(
                    id="trading-timeframe",
                    options=[
                        {"label": "15 Minutes", "value": "15m"},
                        {"label": "30 Minutes", "value": "30m"},
                        {"label": "60 Minutes", "value": "60m"},
                        {"label": "120 Minutes", "value": "120m"}
                    ],
                    value="30m",
                    className="timeframe-options"
                ),
                html.Button("Apply Timeframe", id="apply-timeframe-button", className="apply-button")
            ], className="sidebar-section"),
            
            # Market Overview section
            html.Div([
                html.H3("Market Overview"),
                dash_table.DataTable(
                    id="market-overview-table",
                    columns=[
                        {"name": "Index", "id": "index"},
                        {"name": "Price", "id": "price"},
                        {"name": "Change", "id": "change"}
                    ],
                    data=[
                        {"index": "SPY", "price": "$0.00", "change": "+0.00 (+0.00%)"},
                        {"index": "QQQ", "price": "$0.00", "change": "+0.00 (+0.00%)"},
                        {"index": "IWM", "price": "$0.00", "change": "+0.00 (+0.00%)"},
                        {"index": "DIA", "price": "$0.00", "change": "+0.00 (+0.00%)"}
                    ],
                    style_cell={'textAlign': 'left'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{change} contains "+"',
                            },
                            'color': 'green'
                        },
                        {
                            'if': {
                                'filter_query': '{change} contains "-"',
                            },
                            'color': 'red'
                        }
                    ]
                )
            ], className="sidebar-section"),
            
            # Watchlist section
            html.Div([
                html.H3("Watchlist"),
                html.Div(id="watchlist-container", children=[
                    html.P("Your watchlist is empty")
                ])
            ], className="sidebar-section")
        ], className="sidebar"),
        
        # Main content
        html.Div([
            # Top Recommendations section
            html.Div([
                html.Div([
                    html.H2("Top Recommendations"),
                    # Filter buttons
                    html.Div([
                        html.Button("All", id="filter-all", className="filter-button active"),
                        html.Button("Calls", id="filter-calls", className="filter-button"),
                        html.Button("Puts", id="filter-puts", className="filter-button"),
                        html.Button("Settings", id="filter-settings", className="filter-button settings")
                    ], className="recommendation-filters")
                ], className="recommendations-header"),
                
                # Recommendations cards container
                html.Div(id="top-recommendations-container", className="recommendations-container")
            ], className="main-section"),
            
            # Recommendation Validation section
            html.Div([
                html.H3("Recommendation Validation"),
                html.Div(id="validation-status", className="validation-status"),
                html.Div(id="validation-details", className="validation-details")
            ], className="main-section"),
            
            # Additional Features section
            html.Div([
                html.H3("Additional Features"),
                dcc.Tabs(id="feature-tabs", value="options-chain", children=[
                    dcc.Tab(label="Options Chain", value="options-chain", className="feature-tab"),
                    dcc.Tab(label="Greeks", value="greeks", className="feature-tab"),
                    dcc.Tab(label="Technical Indicators", value="technical-indicators", className="feature-tab"),
                    dcc.Tab(label="Historical Data", value="historical-data", className="feature-tab"),
                    dcc.Tab(label="Real-Time Data", value="real-time-data", className="feature-tab")
                ]),
                html.Div(id="feature-content", className="feature-content")
            ], className="main-section")
        ], className="main-content")
    ], className="app-body"),
    
    # Store components for sharing data between callbacks
    dcc.Store(id="options-data-store"),
    dcc.Store(id="historical-data-store"),
    dcc.Store(id="recommendations-store"),
    
    # Interval for periodic updates
    dcc.Interval(
        id="interval-component",
        interval=60*1000,  # 60 seconds
        n_intervals=0
    )
], className="app-container")

# Callback to update options data store when symbol changes
@app.callback(
    Output("options-data-store", "data"),
    [Input("global-search-button", "n_clicks")],
    [State("global-symbol-input", "value")]
)
def update_options_data(n_clicks, symbol):
    if not symbol:
        return {}
    
    # Get option chain data
    option_chain = data_retriever.get_option_chain(symbol)
    
    return option_chain

# Callback to update historical data store when symbol changes
@app.callback(
    Output("historical-data-store", "data"),
    [Input("global-search-button", "n_clicks")],
    [State("global-symbol-input", "value")]
)
def update_historical_data(n_clicks, symbol):
    if not symbol:
        return {}
    
    # Get historical data
    historical_data = data_retriever.get_historical_data(
        symbol=symbol,
        period_type='day',
        period=10,
        frequency_type='minute',
        frequency=5
    )
    
    # Convert to JSON serializable format
    if not historical_data.empty:
        historical_data_dict = {
            "datetime": historical_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "open": historical_data['open'].tolist(),
            "high": historical_data['high'].tolist(),
            "low": historical_data['low'].tolist(),
            "close": historical_data['close'].tolist(),
            "volume": historical_data['volume'].tolist()
        }
        return historical_data_dict
    
    return {}

# Callback to generate recommendations when symbol changes
@app.callback(
    Output("recommendations-store", "data"),
    [Input("global-search-button", "n_clicks")],
    [State("global-symbol-input", "value")]
)
def generate_recommendations(n_clicks, symbol):
    if not symbol:
        return []
    
    try:
        print(f"Generating recommendations for symbol: {symbol}")
        # Get recommendations from the engine
        recommendations = recommendation_engine.generate_recommendations(symbol)
        
        # Debug information
        print(f"Recommendations type: {type(recommendations)}")
        if isinstance(recommendations, pd.DataFrame):
            print(f"DataFrame shape: {recommendations.shape}")
            print(f"DataFrame columns: {recommendations.columns.tolist() if not recommendations.empty else 'Empty DataFrame'}")
            print(f"First row: {recommendations.iloc[0].to_dict() if not recommendations.empty and len(recommendations) > 0 else 'No data'}")
        else:
            print(f"Recommendations length: {len(recommendations) if recommendations else 0}")
        
        # If recommendations is None or empty, return empty list
        # Fix for DataFrame truth value ambiguity error
        if isinstance(recommendations, pd.DataFrame):
            if recommendations.empty:
                print("Recommendations DataFrame is empty, returning empty list")
                return []
            # Convert DataFrame to list of dictionaries for proper JSON serialization
            result = recommendations.to_dict('records')
            print(f"Converted DataFrame to {len(result)} records")
            return result
        elif not recommendations:
            print("Recommendations is empty or None, returning empty list")
            return []
        
        print(f"Returning {len(recommendations)} recommendations")
        return recommendations
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

# Callback to update top recommendations container
@app.callback(
    Output("top-recommendations-container", "children"),
    [Input("recommendations-store", "data"),
     Input("filter-all", "n_clicks"),
     Input("filter-calls", "n_clicks"),
     Input("filter-puts", "n_clicks")]
)
def update_top_recommendations(recommendations, all_clicks, calls_clicks, puts_clicks):
    print(f"\n=== UPDATE TOP RECOMMENDATIONS DEBUG ===")
    print(f"Recommendations type: {type(recommendations)}")
    print(f"Recommendations data: {recommendations[:2] if recommendations and not isinstance(recommendations, pd.DataFrame) else 'DataFrame or None'}")
    
    # Determine which filter button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        filter_type = "ALL"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "filter-calls":
            filter_type = "CALL"
        elif button_id == "filter-puts":
            filter_type = "PUT"
        else:
            filter_type = "ALL"
    
    print(f"Selected filter type: {filter_type}")
    
    # Filter recommendations based on option type
    if not recommendations:
        print("No recommendations available")
        return html.Div("No recommendations available. Try a different symbol.", className="no-recommendations")
    
    # Fix for DataFrame truth value ambiguity error
    # Ensure recommendations is a list of dictionaries, not a DataFrame
    filtered_recommendations = recommendations
    if filter_type != "ALL":
        try:
            # Safe list comprehension that works with both DataFrame and list of dicts
            if isinstance(recommendations, pd.DataFrame):
                print(f"Processing recommendations as DataFrame with shape: {recommendations.shape}")
                # Filter DataFrame by column value
                filtered_recommendations = recommendations[recommendations['optionType'].str.upper() == filter_type]
                print(f"Filtered DataFrame shape: {filtered_recommendations.shape}")
                # Convert to list of dictionaries for consistency
                filtered_recommendations = filtered_recommendations.to_dict('records')
                print(f"Converted to {len(filtered_recommendations)} records")
            else:
                print(f"Processing recommendations as list with length: {len(recommendations)}")
                # Process as list of dictionaries
                filtered_recommendations = [r for r in recommendations if r.get('optionType', '').upper() == filter_type]
                print(f"Filtered to {len(filtered_recommendations)} records")
        except Exception as e:
            print(f"Error filtering recommendations: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Fall back to original recommendations if filtering fails
            filtered_recommendations = recommendations
    
    # Create trade cards for the recommendations
    return create_trade_cards_container(filtered_recommendations)

# Callback to update feature content based on selected tab
@app.callback(
    Output("feature-content", "children"),
    [Input("feature-tabs", "value"),
     Input("options-data-store", "data"),
     Input("historical-data-store", "data")]
)
def update_feature_content(tab, options_data, historical_data):
    if tab == "options-chain":
        # Display options chain
        if not options_data or not options_data.get('options'):
            return html.Div("No options data available. Enter a symbol and click Search.", className="no-data-message")
        
        # Create options chain table
        options = options_data.get('options', [])
        
        # Group by expiration date
        expirations = options_data.get('expirations', [])
        
        if not expirations:
            return html.Div("No expiration dates available.", className="no-data-message")
        
        # Use the first expiration date
        exp_date = expirations[0]
        
        # Filter options for this expiration
        exp_options = [opt for opt in options if opt['expiration'] == exp_date]
        
        # Create table
        return html.Div([
            html.H4(f"Options Chain for {options_data.get('symbol', '')} - {exp_date}"),
            dash_table.DataTable(
                columns=[
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
                ],
                data=exp_options,
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{option_type} = "CALL"',
                        },
                        'backgroundColor': 'rgba(0, 128, 0, 0.1)'
                    },
                    {
                        'if': {
                            'filter_query': '{option_type} = "PUT"',
                        },
                        'backgroundColor': 'rgba(255, 0, 0, 0.1)'
                    }
                ]
            )
        ])
    
    elif tab == "greeks":
        # Display Greeks visualization
        if not options_data or not options_data.get('options'):
            return html.Div("No options data available. Enter a symbol and click Search.", className="no-data-message")
        
        # Create placeholder for Greeks visualization
        return html.Div([
            html.H4(f"Greeks Analysis for {options_data.get('symbol', '')}"),
            html.P("Greeks visualization will be displayed here.")
        ])
    
    elif tab == "technical-indicators":
        # Display technical indicators
        if not historical_data:
            return html.Div("No historical data available. Enter a symbol and click Search.", className="no-data-message")
        
        # Create placeholder for technical indicators
        return html.Div([
            html.H4("Technical Indicators"),
            html.P("Technical indicators visualization will be displayed here.")
        ])
    
    elif tab == "historical-data":
        # Display historical data
        if not historical_data:
            return html.Div("No historical data available. Enter a symbol and click Search.", className="no-data-message")
        
        # Create placeholder for historical data
        return html.Div([
            html.H4("Historical Data"),
            html.P("Historical data visualization will be displayed here.")
        ])
    
    elif tab == "real-time-data":
        # Display real-time data
        return html.Div([
            html.H4("Real-Time Data"),
            html.P("Real-time data visualization will be displayed here.")
        ])
    
    return html.Div("Select a feature tab to view content.")

# Callback to update authentication status
@app.callback(
    [Output("auth-message", "children"),
     Output("auth-details", "style")],
    [Input("interval-component", "n_intervals")]
)
def update_auth_status(n_intervals):
    # Check if tokens exist
    if os.path.exists(TOKENS_FILE) and os.path.getsize(TOKENS_FILE) > 0:
        return "Authentication successful", {"display": "none"}
    else:
        return "Authentication required. Please check console for instructions.", {"display": "block"}

# Callback to update market overview
@app.callback(
    Output("market-overview-table", "data"),
    [Input("interval-component", "n_intervals")]
)
def update_market_overview(n_intervals):
    # This would normally fetch real market data
    # For now, we'll use placeholder data with random changes
    indices = ["SPY", "QQQ", "IWM", "DIA"]
    prices = [420.50, 380.25, 210.75, 350.30]
    changes = []
    
    for i in range(len(indices)):
        # Generate random change
        change_pct = random.uniform(-0.5, 0.5)
        change_val = prices[i] * change_pct / 100
        
        # Format change string
        if change_val >= 0:
            change_str = f"+{change_val:.2f} (+{change_pct:.2f}%)"
        else:
            change_str = f"{change_val:.2f} ({change_pct:.2f}%)"
        
        changes.append(change_str)
    
    # Create table data
    data = [
        {"index": indices[i], "price": f"${prices[i]:.2f}", "change": changes[i]}
        for i in range(len(indices))
    ]
    
    return data

# Callback to update validation status
@app.callback(
    [Output("validation-status", "children"),
     Output("validation-details", "children")],
    [Input("recommendations-store", "data")]
)
def update_validation_status(recommendations):
    if not recommendations:
        return html.Span("Pending", className="validation-pending"), "No recommendations to validate."
    
    # Simulate validation check
    return html.Span("Passed", className="validation-passed"), "All recommendations have been validated."

# Run the app
if __name__ == '__main__':
    # Open browser automatically
    webbrowser.open('http://localhost:8050')
    
    # Run the app
    app.run_server(debug=True, port=8050)
