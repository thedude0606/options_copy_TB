import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import os
from dotenv import load_dotenv
from app.options_data import OptionsDataRetriever

# Load environment variables
load_dotenv()

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Initialize the options data retriever
options_data = OptionsDataRetriever(interactive_auth=True)

# App layout
app.layout = html.Div([
    html.H1("Schwab Options Dashboard"),
    
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

# Callback to fetch data when symbol is submitted
@app.callback(
    [Output("options-data", "data"),
     Output("historical-data", "data"),
     Output("quote-data", "data"),
     Output("expiration-dropdown", "options")],
    [Input("submit-button", "n_clicks")],
    [State("symbol-input", "value")]
)
def fetch_data(n_clicks, symbol):
    if n_clicks == 0:
        # Default data for initial load
        return None, None, None, []
    
    if not symbol:
        return None, None, None, []
    
    # Get option chain
    option_chain = options_data.get_option_chain(symbol)
    
    # Get historical data
    historical_data = options_data.get_historical_data(symbol)
    
    # Get current quote
    quote = options_data.get_quote(symbol)
    
    # Extract expiration dates for dropdown
    expiration_dates = []
    if option_chain:
        # This will need to be adapted based on the actual API response structure
        # Placeholder for expiration dates extraction
        expiration_dates = [{"label": "2023-04-21", "value": "2023-04-21"}]
    
    return option_chain, historical_data, quote, expiration_dates

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
    # This will need to be adapted based on the actual data structure
    
    # Create table to display options data
    return html.Div("Options chain will be displayed here")

# Callback to update Greeks visualization
@app.callback(
    Output("greeks-container", "children"),
    [Input("options-data", "data"),
     Input("expiration-dropdown", "value")]
)
def update_greeks(options_data, expiration):
    if not options_data or not expiration:
        return html.Div("No data available. Please enter a symbol and select an expiration date.")
    
    # Create visualizations for Greeks
    return html.Div("Greeks visualization will be displayed here")

# Callback to update historical chart
@app.callback(
    Output("historical-chart", "figure"),
    [Input("historical-data", "data"),
     Input("time-period", "value")]
)
def update_historical_chart(historical_data, time_period):
    if not historical_data:
        return go.Figure()
    
    # Create candlestick chart
    fig = go.Figure()
    
    # This will need to be adapted based on the actual data structure
    fig.add_trace(go.Candlestick(
        x=[1, 2, 3, 4, 5],
        open=[10, 11, 12, 11, 10],
        high=[12, 13, 14, 13, 12],
        low=[9, 10, 11, 10, 9],
        close=[11, 12, 13, 12, 11],
        name="Price"
    ))
    
    fig.update_layout(
        title=f"Historical Price Data",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
