"""
Greeks analysis tab for options recommendation platform.
Implements the UI for displaying and analyzing options Greeks.
"""
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from app.analysis.options_analysis import OptionsAnalysis
from app.data_collector import DataCollector

def create_greeks_tab(symbol=None):
    """
    Create the Greeks analysis tab layout
    
    Args:
        symbol (str, optional): Stock symbol to pre-populate
    
    Returns:
        html.Div: Greeks analysis tab layout
    """
    return html.Div([
        dbc.Row([
            # Controls column
            dbc.Col([
                html.H4("Options Greeks Analysis", className="mb-3"),
                html.Label("Symbol:"),
                dbc.InputGroup([
                    dbc.Input(id="greeks-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
                    dbc.Button("Analyze", id="greeks-analyze-button", color="primary")
                ], className="mb-3"),
                
                # Expiration date selection
                html.Label("Expiration Date:"),
                dcc.Dropdown(id="greeks-expiration-dropdown", placeholder="Select expiration date", className="mb-3"),
                
                # Option type selection
                html.Label("Option Type:"),
                dbc.RadioItems(
                    id="greeks-option-type",
                    options=[
                        {"label": "Calls", "value": "CALL"},
                        {"label": "Puts", "value": "PUT"}
                    ],
                    value="CALL",
                    inline=True,
                    className="mb-3"
                ),
                
                # Greek selection
                html.Label("Greek to Visualize:"),
                dbc.RadioItems(
                    id="greek-selection",
                    options=[
                        {"label": "Delta", "value": "delta"},
                        {"label": "Gamma", "value": "gamma"},
                        {"label": "Theta", "value": "theta"},
                        {"label": "Vega", "value": "vega"},
                        {"label": "Rho", "value": "rho"}
                    ],
                    value="delta",
                    className="mb-3"
                ),
                
                # Greek explanation
                html.Div([
                    html.H5("Greek Explanation", className="mb-2"),
                    html.Div(id="greek-explanation", className="mb-3")
                ], style={"background-color": "#f8f9fa", "padding": "10px", "border-radius": "5px"}),
                
                # Filters
                html.H5("Filters", className="mb-2"),
                html.Label("Strike Range:"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Min:"),
                        dbc.Input(id="strike-min", type="number", placeholder="Min")
                    ], width=6),
                    dbc.Col([
                        html.Label("Max:"),
                        dbc.Input(id="strike-max", type="number", placeholder="Max")
                    ], width=6)
                ], className="mb-3"),
                
                # Reset filters button
                dbc.Button(
                    "Reset Filters",
                    id="reset-greeks-filters",
                    color="secondary",
                    size="sm",
                    className="mb-3"
                ),
                
                # Update button
                dbc.Button(
                    "Update Chart",
                    id="update-greeks-button",
                    color="primary",
                    className="mt-2"
                )
            ], width=3),
            
            # Chart column
            dbc.Col([
                html.Div(id="greeks-chart-container", children=[
                    dcc.Loading(
                        id="greeks-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="greeks-chart", style={"height": "800px"})
                        ]
                    )
                ])
            ], width=9)
        ])
    ])

# Callback to reset Greeks filters
def reset_greeks_filters(n_clicks):
    """
    Reset Greeks filters to default values
    
    Args:
        n_clicks (int): Number of clicks
        
    Returns:
        tuple: Default values for filters
    """
    if n_clicks:
        return None, None
    return dash.no_update, dash.no_update

# Callback to update Greek explanation
def update_greeks_explanation(greek):
    """
    Update the explanation for the selected Greek
    
    Args:
        greek (str): Selected Greek
        
    Returns:
        html.Div: Explanation for the selected Greek
    """
    explanations = {
        "delta": [
            "Delta measures the rate of change in an option's price for a $1 change in the underlying asset.",
            "For call options, delta ranges from 0 to 1. For put options, delta ranges from -1 to 0.",
            "Delta can be interpreted as the approximate probability of the option expiring in-the-money."
        ],
        "gamma": [
            "Gamma measures the rate of change in delta for a $1 change in the underlying asset.",
            "It is a second-order derivative and represents the curvature of the option's value.",
            "Higher gamma means the option's delta will change more rapidly with movements in the underlying."
        ],
        "theta": [
            "Theta measures the rate of time decay in an option's value per day.",
            "It is typically negative for both calls and puts, as options lose value as time passes.",
            "Theta accelerates as expiration approaches, especially for at-the-money options."
        ],
        "vega": [
            "Vega measures the rate of change in an option's price for a 1% change in implied volatility.",
            "Higher vega means the option is more sensitive to changes in volatility.",
            "Vega is typically highest for at-the-money options with longer time to expiration."
        ],
        "rho": [
            "Rho measures the rate of change in an option's price for a 1% change in interest rates.",
            "Call options typically have positive rho, while put options typically have negative rho.",
            "Rho's effect is usually minimal for short-term options but more significant for longer-term options."
        ]
    }
    
    explanation = explanations.get(greek, ["No explanation available for this Greek."])
    
    return html.Div([
        html.P(point) for point in explanation
    ])

# Callback to update Greeks chart
def update_greeks_chart(update_clicks, analyze_clicks, 
                       symbol, expiration, option_type, greek,
                       strike_min, strike_max):
    """
    Update the Greeks analysis chart
    
    Args:
        update_clicks (int): Number of update button clicks
        analyze_clicks (int): Number of analyze button clicks
        symbol (str): Stock symbol
        expiration (str): Selected expiration date
        option_type (str): Option type (CALL or PUT)
        greek (str): Selected Greek to visualize
        strike_min (float): Minimum strike price filter
        strike_max (float): Maximum strike price filter
        
    Returns:
        go.Figure: Greeks analysis chart
    """
    # Check if callback was triggered
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure()
    
    # Get trigger ID
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Only update if triggered by buttons
    if trigger_id not in ['update-greeks-button', 'greeks-analyze-button']:
        return dash.no_update
    
    # Validate symbol and expiration
    if not symbol or not expiration:
        return go.Figure()
    
    try:
        # Get options data
        data_collector = DataCollector()
        options_data = data_collector.get_option_data(symbol, option_type=option_type)
        
        if options_data.empty:
            return go.Figure()
        
        # Filter by expiration date
        matching_options = options_data[
            (options_data['expirationDate'] == pd.to_datetime(expiration))
        ]
        
        if matching_options.empty:
            return go.Figure()
        
        # Apply strike filters if provided
        if strike_min is not None:
            matching_options = matching_options[matching_options['strikePrice'] >= strike_min]
        
        if strike_max is not None:
            matching_options = matching_options[matching_options['strikePrice'] <= strike_max]
        
        if matching_options.empty:
            return go.Figure()
        
        # Sort by strike price
        matching_options = matching_options.sort_values('strikePrice')
        
        # Create 3D surface plot for the selected Greek
        if greek in matching_options.columns:
            # Get unique strike prices and days to expiration
            strikes = matching_options['strikePrice'].unique()
            
            # Use days_numeric instead of .dt.days
            def get_days(x):
                if isinstance(x, pd.Timedelta):
                    return x.days
                elif pd.isna(x):
                    return 0
                else:
                    return float(x)
            
            # Create a grid for the 3D surface
            strike_grid = []
            days_grid = []
            greek_values = []
            
            # Populate the grid
            for strike in strikes:
                strike_options = matching_options[matching_options['strikePrice'] == strike]
                
                for _, row in strike_options.iterrows():
                    days = get_days(row['daysToExpiration'])
                    strike_grid.append(strike)
                    days_grid.append(days)
                    greek_values.append(row[greek])
            
            # Create the 3D surface plot
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=strike_grid,
                    y=days_grid,
                    z=greek_values,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=greek_values,
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title=greek.capitalize())
                    ),
                    name=greek.capitalize()
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=f"{greek.capitalize()} Analysis for {symbol} {option_type}s (Expiration: {expiration})",
                scene=dict(
                    xaxis_title="Strike Price",
                    yaxis_title="Days to Expiration",
                    zaxis_title=greek.capitalize(),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=800
            )
            
            return fig
        else:
            # Create 2D plot if the Greek is not available
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=matching_options['strikePrice'],
                    y=matching_options['last'],
                    mode='markers',
                    name='Option Price'
                )
            )
            
            fig.update_layout(
                title=f"Option Prices for {symbol} {option_type}s (Expiration: {expiration})",
                xaxis_title="Strike Price",
                yaxis_title="Option Price",
                height=800
            )
            
            return fig
    
    except Exception as e:
        print(f"Error updating Greeks chart: {str(e)}")
        return go.Figure()

def register_greeks_callbacks(app):
    """
    Register callback functions for the Greeks tab
    
    Args:
        app: The Dash app instance
    """
    # Callback to reset Greeks filters
    @app.callback(
        [Output("strike-min", "value"),
         Output("strike-max", "value")],
        [Input("reset-greeks-filters", "n_clicks")]
    )
    def reset_filters_callback(n_clicks):
        return reset_greeks_filters(n_clicks)
    
    # Callback to update Greek explanation
    @app.callback(
        Output("greek-explanation", "children"),
        [Input("greek-selection", "value")]
    )
    def update_explanation_callback(greek):
        return update_greeks_explanation(greek)
    
    # Callback to populate expiration dropdown
    @app.callback(
        Output("greeks-expiration-dropdown", "options", allow_duplicate=True),
        [Input("greeks-analyze-button", "n_clicks")],
        [State("greeks-symbol-input", "value")],
        prevent_initial_call=True
    )
    def update_expiration_options(n_clicks, symbol):
        if not n_clicks or not symbol:
            return []
        
        try:
            # Get options data
            data_collector = DataCollector()
            options_data = data_collector.get_option_data(symbol)
            
            # Check if options_data is a DataFrame or a dictionary
            if isinstance(options_data, pd.DataFrame):
                if options_data.empty:
                    return []
            elif isinstance(options_data, dict):
                if not options_data:  # Check if dictionary is empty
                    return []
            else:
                # If it's neither a DataFrame nor a dictionary, or is None
                if options_data is None:
                    return []
                return []  # Return empty list for any other type
            
            # Get unique expiration dates
            expirations = options_data['expirationDate'].unique()
            
            # Format as options for dropdown
            options = [{"label": exp.strftime('%Y-%m-%d'), "value": exp.strftime('%Y-%m-%d')} for exp in expirations]
            
            return options
        except Exception as e:
            print(f"Error updating expiration options: {str(e)}")
            return []
    
    # Callback to update Greeks chart
    @app.callback(
        Output("greeks-chart", "figure"),
        [Input("update-greeks-button", "n_clicks"),
         Input("greeks-analyze-button", "n_clicks")],
        [State("greeks-symbol-input", "value"),
         State("greeks-expiration-dropdown", "value"),
         State("greeks-option-type", "value"),
         State("greek-selection", "value"),
         State("strike-min", "value"),
         State("strike-max", "value")]
    )
    def update_chart_callback(update_clicks, analyze_clicks, 
                             symbol, expiration, option_type, greek,
                             strike_min, strike_max):
        return update_greeks_chart(update_clicks, analyze_clicks, 
                                  symbol, expiration, option_type, greek,
                                  strike_min, strike_max)
