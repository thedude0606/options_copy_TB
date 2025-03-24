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

def create_greeks_tab():
    """
    Create the Greeks analysis tab layout
    
    Returns:
        html.Div: Greeks analysis tab layout
    """
    return html.Div([
        dbc.Row([
            # Symbol input and controls
            dbc.Col([
                html.H4("Options Greeks Analysis", className="mb-3"),
                html.Label("Symbol:"),
                dbc.InputGroup([
                    dbc.Input(id="greeks-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
                    dbc.Button("Analyze", id="greeks-analyze-button", color="primary")
                ], className="mb-3"),
                
                # Option type selection
                html.Label("Option Type:"),
                dbc.RadioItems(
                    id="greeks-option-type",
                    options=[
                        {"label": "Calls", "value": "CALL"},
                        {"label": "Puts", "value": "PUT"},
                        {"label": "Both", "value": "ALL"}
                    ],
                    value="ALL",
                    inline=True,
                    className="mb-3"
                ),
                
                # Expiration selection
                html.Label("Expiration Range:"),
                dbc.Select(
                    id="greeks-expiration-range",
                    options=[
                        {"label": "Weekly (< 7 days)", "value": "weekly"},
                        {"label": "Short-term (7-30 days)", "value": "short"},
                        {"label": "Medium-term (30-90 days)", "value": "medium"},
                        {"label": "Long-term (> 90 days)", "value": "long"},
                        {"label": "All Expirations", "value": "all"}
                    ],
                    value="all",
                    className="mb-3"
                ),
                
                # Strike range selection
                html.Label("Strike Range:"),
                dbc.Select(
                    id="greeks-strike-range",
                    options=[
                        {"label": "In-the-Money", "value": "itm"},
                        {"label": "At-the-Money", "value": "atm"},
                        {"label": "Out-of-the-Money", "value": "otm"},
                        {"label": "All Strikes", "value": "all"}
                    ],
                    value="all",
                    className="mb-3"
                ),
                
                # Greek visualization selection
                html.H5("Greek Visualization", className="mb-2"),
                dbc.RadioItems(
                    id="greeks-visualization",
                    options=[
                        {"label": "Delta", "value": "delta"},
                        {"label": "Gamma", "value": "gamma"},
                        {"label": "Theta", "value": "theta"},
                        {"label": "Vega", "value": "vega"},
                        {"label": "Rho", "value": "rho"},
                        {"label": "All Greeks", "value": "all"}
                    ],
                    value="delta",
                    className="mb-3"
                ),
                
                # Visualization type
                html.Label("Visualization Type:"),
                dbc.RadioItems(
                    id="greeks-visualization-type",
                    options=[
                        {"label": "Surface Plot", "value": "surface"},
                        {"label": "Heatmap", "value": "heatmap"},
                        {"label": "Line Chart", "value": "line"}
                    ],
                    value="surface",
                    inline=True,
                    className="mb-3"
                ),
                
                # Action buttons
                dbc.Button("Update Chart", id="update-greeks-button", color="success", className="mr-2 mt-3"),
                dbc.Button("Reset Filters", id="reset-greeks-button", color="secondary", outline=True, className="mt-3")
            ], md=3, className="sidebar"),
            
            # Chart display area
            dbc.Col([
                # Loading spinner for chart
                dbc.Spinner(
                    dcc.Graph(id="greeks-chart", style={"height": "700px"}),
                    color="primary",
                    type="border",
                    fullscreen=False
                ),
                
                # Greeks explanation
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Understanding Options Greeks", className="card-title"),
                        html.Div(id="greeks-explanation", className="mt-2")
                    ]),
                    className="mt-3"
                ),
                
                # Status messages
                html.Div(id="greeks-status", className="status-message mt-3")
            ], md=9)
        ])
    ])

# Callback for resetting filters
@callback(
    [Output("greeks-option-type", "value"),
     Output("greeks-expiration-range", "value"),
     Output("greeks-strike-range", "value"),
     Output("greeks-visualization", "value"),
     Output("greeks-visualization-type", "value")],
    Input("reset-greeks-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_greeks_filters(n_clicks):
    return "ALL", "all", "all", "delta", "surface"

# Callback for updating Greeks explanation
@callback(
    Output("greeks-explanation", "children"),
    Input("greeks-visualization", "value"),
    prevent_initial_call=True
)
def update_greeks_explanation(greek):
    explanations = {
        "delta": [
            html.P("Delta measures the rate of change in an option's price relative to a $1 change in the underlying asset."),
            html.P("For call options, delta ranges from 0 to 1. For put options, delta ranges from -1 to 0."),
            html.P("Delta can also be interpreted as the approximate probability of the option expiring in-the-money.")
        ],
        "gamma": [
            html.P("Gamma measures the rate of change in delta relative to a $1 change in the underlying asset."),
            html.P("It is highest for at-the-money options and decreases as options move in- or out-of-the-money."),
            html.P("High gamma means the option's delta will change rapidly with small movements in the underlying.")
        ],
        "theta": [
            html.P("Theta measures the rate of time decay in an option's value, representing the daily loss in option value."),
            html.P("It is typically negative for both calls and puts, as options lose value as time passes."),
            html.P("Theta accelerates as expiration approaches, especially for at-the-money options.")
        ],
        "vega": [
            html.P("Vega measures the rate of change in an option's price relative to a 1% change in implied volatility."),
            html.P("Higher vega means the option is more sensitive to changes in volatility."),
            html.P("Vega is highest for at-the-money options with longer time to expiration.")
        ],
        "rho": [
            html.P("Rho measures the rate of change in an option's price relative to a 1% change in interest rates."),
            html.P("Call options typically have positive rho, while put options have negative rho."),
            html.P("Rho's impact is most significant for longer-term options.")
        ],
        "all": [
            html.P("Options Greeks are a set of risk measures that describe how option prices change with respect to various factors:"),
            html.P("Delta: Change in option price per $1 change in underlying price"),
            html.P("Gamma: Rate of change in delta per $1 change in underlying price"),
            html.P("Theta: Change in option price per day (time decay)"),
            html.P("Vega: Change in option price per 1% change in implied volatility"),
            html.P("Rho: Change in option price per 1% change in interest rates")
        ]
    }
    
    return explanations.get(greek, explanations["all"])

# Callback for updating Greeks chart
@callback(
    [Output("greeks-chart", "figure"),
     Output("greeks-status", "children")],
    [Input("update-greeks-button", "n_clicks"),
     Input("greeks-analyze-button", "n_clicks")],
    [State("greeks-symbol-input", "value"),
     State("greeks-option-type", "value"),
     State("greeks-expiration-range", "value"),
     State("greeks-strike-range", "value"),
     State("greeks-visualization", "value"),
     State("greeks-visualization-type", "value")],
    prevent_initial_call=True
)
def update_greeks_chart(update_clicks, analyze_clicks, 
                       symbol, option_type, expiration_range, 
                       strike_range, greek_type, visualization_type):
    # Check if any button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure(), ""
    
    # Validate inputs
    if not symbol:
        return go.Figure(), html.Div("Please enter a symbol", className="text-danger")
    
    try:
        # Initialize data collector and options analysis
        data_collector = DataCollector(interactive_auth=False)
        options_analysis = OptionsAnalysis()
        
        # Get options data
        options_data = data_collector.get_option_data(symbol.upper())
        
        if options_data.empty:
            return go.Figure(), html.Div(f"No options data available for {symbol.upper()}", className="text-warning")
        
        # Calculate Greeks if not already present
        if not all(g in options_data.columns for g in ['delta', 'gamma', 'theta', 'vega', 'rho']):
            options_data = options_analysis.calculate_all_greeks(options_data)
        
        # Filter by option type
        if option_type != "ALL":
            options_data = options_data[options_data['optionType'] == option_type]
        
        # Filter by expiration range
        if expiration_range != "all":
            # Handle daysToExpiration which could be a Timedelta or a numeric value
            def get_days(x):
                if isinstance(x, pd.Timedelta):
                    return x.days
                elif pd.isna(x):
                    return 0
                else:
                    return float(x)
            
            # Apply the function to create a numeric days column
            options_data['days_numeric'] = options_data['daysToExpiration'].apply(get_days)
            
            if expiration_range == "weekly":
                options_data = options_data[options_data['days_numeric'] < 7]
            elif expiration_range == "short":
                options_data = options_data[(options_data['days_numeric'] >= 7) & (options_data['days_numeric'] <= 30)]
            elif expiration_range == "medium":
                options_data = options_data[(options_data['days_numeric'] > 30) & (options_data['days_numeric'] <= 90)]
            elif expiration_range == "long":
                options_data = options_data[options_data['days_numeric'] > 90]
        
        # Filter by strike range
        if strike_range != "all" and 'underlyingPrice' in options_data.columns:
            underlying_price = options_data['underlyingPrice'].iloc[0]
            
            if strike_range == "itm":
                # ITM: calls with strike < underlying, puts with strike > underlying
                mask = ((options_data['optionType'] == 'CALL') & (options_data['strikePrice'] < underlying_price)) | \
                       ((options_data['optionType'] == 'PUT') & (options_data['strikePrice'] > underlying_price))
                options_data = options_data[mask]
            elif strike_range == "atm":
                # ATM: strike within 5% of underlying
                options_data = options_data[abs(options_data['strikePrice'] - underlying_price) / underlying_price <= 0.05]
            elif strike_range == "otm":
                # OTM: calls with strike > underlying, puts with strike < underlying
                mask = ((options_data['optionType'] == 'CALL') & (options_data['strikePrice'] > underlying_price)) | \
                       ((options_data['optionType'] == 'PUT') & (options_data['strikePrice'] < underlying_price))
                options_data = options_data[mask]
        
        if options_data.empty:
            return go.Figure(), html.Div(f"No options data available for {symbol.upper()} with the selected filters", className="text-warning")
        
        # Prepare data for visualization
        if greek_type == "all":
            # Create subplots for all Greeks
            fig = make_subplots(rows=3, cols=2, subplot_titles=["Delta", "Gamma", "Theta", "Vega", "Rho", ""],
                               specs=[[{"type": "surface"}, {"type": "surface"}],
                                     [{"type": "surface"}, {"type": "surface"}],
                                     [{"type": "surface"}, {"type": "surface"}]])
            
            # Add each Greek to its subplot
            greek_names = ["delta", "gamma", "theta", "vega", "rho"]
            positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
            
            for greek, pos in zip(greek_names, positions):
                if greek in options_data.columns:
                    create_greek_plot(fig, options_data, greek, visualization_type, pos[0], pos[1])
            
            # Update layout
            fig.update_layout(
                title=f"{symbol.upper()} Options Greeks Analysis",
                height=1000,
                margin=dict(l=50, r=50, t=80, b=50)
            )
        else:
            # Create single plot for selected Greek
            fig = go.Figure()
            
            if greek_type in options_data.columns:
                create_greek_plot(fig, options_data, greek_type, visualization_type)
            
            # Update layout
            fig.update_layout(
                title=f"{symbol.upper()} Options {greek_type.capitalize()} Analysis",
                height=700,
                margin=dict(l=50, r=50, t=80, b=50)
            )
        
        return fig, html.Div(f"Greeks analysis for {symbol.upper()} completed successfully", className="text-success")
    
    except Exception as e:
        return go.Figure(), html.Div(f"Error generating Greeks analysis: {str(e)}", className="text-danger")

def create_greek_plot(fig, options_data, greek, plot_type, row=1, col=1):
    """
    Create a plot for a specific Greek
    
    Args:
        fig (go.Figure): Figure to add the plot to
        options_data (pd.DataFrame): Options data
        greek (str): Greek to plot (delta, gamma, theta, vega, rho)
        plot_type (str): Type of plot (surface, heatmap, line)
        row (int): Row in subplot grid
        col (int): Column in subplot grid
    """
    # Get unique strikes and expirations
    strikes = sorted(options_data['strikePrice'].unique())
    
    # Handle daysToExpiration which could be a Timedelta or a numeric value
    def get_days(x):
        if isinstance(x, pd.Timedelta):
            return x.days
        elif pd.isna(x):
            return 0
        else:
            return float(x)
    
    # Apply the function to create a numeric days column if it doesn't exist
    if 'days_numeric' not in options_data.columns:
        options_data['days_numeric'] = options_data['daysToExpiration'].apply(get_days)
    
    expirations = sorted(options_data['days_numeric'].unique())
    
    # Create meshgrid for surface plot
    strike_grid, expiry_grid = np.meshgrid(strikes, expirations)
    
    # Initialize z values
    z_values = np.zeros_like(strike_grid, dtype=float)
    z_values.fill(np.nan)
    
    # Fill z values
    for i, expiry in enumerate(expirations):
        for j, strike in enumerate(strikes):
            matching_options = options_data[
                (options_data['strikePrice'] == strike) & 
                (options_data['days_numeric'] == expiry)
            ]
            
            if not matching_options.empty and greek in matching_options.columns:
                z_values[i, j] = matching_options[greek].mean()
    
    # Create appropriate plot based on type
    if plot_type == "surface":
        fig.add_trace(
            go.Surface(
                x=strike_grid,
                y=expiry_grid,
                z=z_values,
                colorscale="Viridis",
                name=greek.capitalize()
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_scenes(
            xaxis_title="Strike Price",
            yaxis_title="Days to Expiration",
            zaxis_title=greek.capitalize(),
            row=row, col=col
        )
    
    elif plot_type == "heatmap":
        fig.add_trace(
            go.Heatmap(
                x=strikes,
                y=expirations,
                z=z_values,
                colorscale="Viridis",
                name=greek.capitalize()
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Strike Price", row=row, col=col)
        fig.update_yaxes(title_text="Days to Expiration", row=row, col=col)
    
    elif plot_type == "line":
        # Group by strike and get mean values
        grouped_data = options_data.groupby('strikePrice')[greek].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=grouped_data['strikePrice'],
                y=grouped_data[greek],
                mode='lines+markers',
                name=greek.capitalize()
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Strike Price", row=row, col=col)
        fig.update_yaxes(title_text=greek.capitalize(), row=row, col=col)
        
        # Add reference line at y=0
        fig.add_shape(
            type="line",
            x0=min(grouped_data['strikePrice']),
            y0=0,
            x1=max(grouped_data['strikePrice']),
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            row=row, col=col
        )
