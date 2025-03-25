"""
Recommendations tab for options recommendation platform.
Implements the UI for displaying option recommendations.
"""
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import json
from app.components.trade_card import create_trade_cards_container
from app.analysis.recommendation_engine import RecommendationEngine
from app.data_collector import DataCollector

def create_recommendations_tab():
    """
    Create the recommendations tab layout
    
    Returns:
        html.Div: Recommendations tab layout
    """
    return html.Div([
        dbc.Row([
            # Symbol input and search controls
            dbc.Col([
                html.H4("Options Recommendations", className="mb-3"),
                html.Label("Symbol:"),
                dbc.InputGroup([
                    dbc.Input(id="recommendation-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
                    dbc.Button("Search", id="recommendation-search-button", color="primary")
                ], className="mb-3"),
                
                # Filters and settings
                html.Div([
                    html.H5("Filters", className="mb-2"),
                    
                    # Option type filter
                    html.Label("Option Type:"),
                    dbc.RadioItems(
                        id="recommendation-option-type",
                        options=[
                            {"label": "Calls", "value": "CALL"},
                            {"label": "Puts", "value": "PUT"},
                            {"label": "Both", "value": "ALL"}
                        ],
                        value="ALL",
                        inline=True,
                        className="mb-3"
                    ),
                    
                    # Time horizon filter
                    html.Label("Time Horizon:"),
                    dbc.Select(
                        id="recommendation-time-horizon",
                        options=[
                            {"label": "Short-term (< 2 weeks)", "value": "short"},
                            {"label": "Medium-term (2-6 weeks)", "value": "medium"},
                            {"label": "Long-term (> 6 weeks)", "value": "long"}
                        ],
                        value="medium",
                        className="mb-3"
                    ),
                    
                    # Confidence threshold slider
                    html.Label("Minimum Confidence Score:"),
                    dcc.Slider(
                        id="recommendation-confidence-threshold",
                        min=0.5,
                        max=0.9,
                        step=0.05,
                        value=0.6,
                        marks={i/10: f"{i/10:.1f}" for i in range(5, 10)},
                        className="mb-4"
                    ),
                    
                    # Advanced settings (collapsible)
                    dbc.Button(
                        "Advanced Settings",
                        id="advanced-settings-toggle",
                        color="secondary",
                        outline=True,
                        size="sm",
                        className="mb-2"
                    ),
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody([
                                # Risk-reward threshold
                                html.Label("Minimum Risk-Reward Ratio:"),
                                dbc.Input(
                                    id="recommendation-risk-reward-threshold",
                                    type="number",
                                    min=1,
                                    max=5,
                                    step=0.5,
                                    value=1.5,
                                    className="mb-2"
                                ),
                                
                                # Probability threshold
                                html.Label("Minimum Probability of Profit:"),
                                dbc.Input(
                                    id="recommendation-probability-threshold",
                                    type="number",
                                    min=0.3,
                                    max=0.8,
                                    step=0.05,
                                    value=0.4,
                                    className="mb-2"
                                ),
                                
                                # Technical indicators weights
                                html.Label("Technical Indicators Weights:"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("RSI:"),
                                        dbc.Input(
                                            id="weight-rsi",
                                            type="number",
                                            min=0,
                                            max=10,
                                            step=1,
                                            value=5
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("MACD:"),
                                        dbc.Input(
                                            id="weight-macd",
                                            type="number",
                                            min=0,
                                            max=10,
                                            step=1,
                                            value=5
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("BB:"),
                                        dbc.Input(
                                            id="weight-bb",
                                            type="number",
                                            min=0,
                                            max=10,
                                            step=1,
                                            value=5
                                        )
                                    ], width=4)
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("IMI:"),
                                        dbc.Input(
                                            id="weight-imi",
                                            type="number",
                                            min=0,
                                            max=10,
                                            step=1,
                                            value=3
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("MFI:"),
                                        dbc.Input(
                                            id="weight-mfi",
                                            type="number",
                                            min=0,
                                            max=10,
                                            step=1,
                                            value=3
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("FVG:"),
                                        dbc.Input(
                                            id="weight-fvg",
                                            type="number",
                                            min=0,
                                            max=10,
                                            step=1,
                                            value=4
                                        )
                                    ], width=4)
                                ])
                            ])
                        ),
                        id="advanced-settings-collapse",
                        is_open=False
                    )
                ], className="filter-container mb-4"),
                
                # Action buttons
                dbc.Button("Generate Recommendations", id="generate-recommendations-button", color="success", className="mr-2"),
                dbc.Button("Reset Filters", id="reset-filters-button", color="secondary", outline=True)
            ], md=3, className="sidebar"),
            
            # Recommendations display area
            dbc.Col([
                # Loading spinner for recommendations
                dbc.Spinner(
                    html.Div(id="recommendations-container", className="recommendations-container"),
                    color="primary",
                    type="border",
                    fullscreen=False
                ),
                
                # Status messages
                html.Div(id="recommendations-status", className="status-message mt-3")
            ], md=9)
        ])
    ])

# Callback for toggling advanced settings
@callback(
    Output("advanced-settings-collapse", "is_open"),
    Input("advanced-settings-toggle", "n_clicks"),
    State("advanced-settings-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_advanced_settings(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback for resetting filters
@callback(
    [Output("recommendation-option-type", "value"),
     Output("recommendation-time-horizon", "value"),
     Output("recommendation-confidence-threshold", "value"),
     Output("recommendation-risk-reward-threshold", "value"),
     Output("recommendation-probability-threshold", "value"),
     Output("weight-rsi", "value"),
     Output("weight-macd", "value"),
     Output("weight-bb", "value"),
     Output("weight-imi", "value"),
     Output("weight-mfi", "value"),
     Output("weight-fvg", "value")],
    Input("reset-filters-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return "ALL", "medium", 0.6, 1.5, 0.4, 5, 5, 5, 3, 3, 4

# Callback for generating recommendations
@callback(
    [Output("recommendations-container", "children"),
     Output("recommendations-status", "children")],
    [Input("generate-recommendations-button", "n_clicks"),
     Input("recommendation-search-button", "n_clicks")],
    [State("recommendation-symbol-input", "value"),
     State("recommendation-option-type", "value"),
     State("recommendation-time-horizon", "value"),
     State("recommendation-confidence-threshold", "value"),
     State("recommendation-risk-reward-threshold", "value"),
     State("recommendation-probability-threshold", "value"),
     State("weight-rsi", "value"),
     State("weight-macd", "value"),
     State("weight-bb", "value"),
     State("weight-imi", "value"),
     State("weight-mfi", "value"),
     State("weight-fvg", "value")],
    prevent_initial_call=True
)
def generate_recommendations(generate_clicks, search_clicks, 
                            symbol, option_type, time_horizon, 
                            confidence_threshold, risk_reward_threshold, 
                            probability_threshold, weight_rsi, weight_macd, 
                            weight_bb, weight_imi, weight_mfi, weight_fvg):
    # Check if any button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div(), ""
    
    # Validate inputs
    if not symbol:
        return html.Div(), html.Div("Please enter a symbol", className="text-danger")
    
    try:
        # Initialize data collector and recommendation engine
        data_collector = DataCollector(interactive_auth=False)
        recommendation_engine = RecommendationEngine(data_collector)
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(
            symbol=symbol.upper(),
            lookback_days=30,
            confidence_threshold=confidence_threshold
        )
        
        # Filter recommendations based on user criteria
        if not recommendations.empty:
            # Filter by option type
            if option_type != "ALL":
                recommendations = recommendations[recommendations['optionType'] == option_type]
            
            # Filter by time horizon
            # Handle daysToExpiration which could be a Timedelta or a numeric value
            def get_days(x):
                if isinstance(x, pd.Timedelta):
                    return x.days
                elif pd.isna(x):
                    return 0
                else:
                    return float(x)
                
            # Apply the function to create a numeric days column
            if 'daysToExpiration' in recommendations.columns:
                recommendations['days_numeric'] = recommendations['daysToExpiration'].apply(get_days)
                
                if time_horizon == "short":
                    # Short-term: < 14 days
                    recommendations = recommendations[recommendations['days_numeric'] < 14]
                elif time_horizon == "medium":
                    # Medium-term: 14-42 days
                    recommendations = recommendations[(recommendations['days_numeric'] >= 14) & 
                                                    (recommendations['days_numeric'] <= 42)]
                else:  # long
                    # Long-term: > 42 days
                    recommendations = recommendations[recommendations['days_numeric'] > 42]
            
            # Filter by risk-reward threshold
            if 'riskRewardRatio' in recommendations.columns:
                recommendations = recommendations[recommendations['riskRewardRatio'] >= risk_reward_threshold]
            
            # Filter by probability threshold
            if 'probabilityOfProfit' in recommendations.columns:
                recommendations = recommendations[recommendations['probabilityOfProfit'] >= probability_threshold]
            
            # Convert to list of dictionaries for trade cards
            recommendations_list = recommendations.to_dict('records')
            
            # Create trade cards
            if recommendations_list:
                return create_trade_cards_container(recommendations_list), html.Div(f"Found {len(recommendations_list)} recommendations for {symbol.upper()}", className="text-success")
            else:
                return html.Div(), html.Div(f"No recommendations found for {symbol.upper()} with the current filters. Try adjusting your criteria.", className="text-warning")
        else:
            return html.Div(), html.Div(f"No recommendations available for {symbol.upper()}. Try another symbol.", className="text-warning")
    
    except Exception as e:
        return html.Div(), html.Div(f"Error generating recommendations: {str(e)}", className="text-danger")

def register_recommendations_callbacks(app, recommendation_engine):
    """
    Register all callbacks for the recommendations tab
    
    Args:
        app: The Dash app
        recommendation_engine: The recommendation engine instance
    """
    # The callbacks are registered using the @callback decorator
    # This function is needed to ensure the callbacks are registered with the app
    # and to provide access to the recommendation_engine
    
    # In a more complex implementation, you might register callbacks programmatically here
    # For now, we're just ensuring this function exists for the import in run_dashboard.py
    
    # You could add additional initialization here if needed
    print("Recommendations tab callbacks registered")
    return
