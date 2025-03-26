"""
Simplified dashboard layout for options recommendation platform.
Focuses on recommendations while maintaining access to other features.
"""
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta

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

def create_simplified_layout(app, options_analysis):
    """
    Create a simplified dashboard layout focused on recommendations
    
    Args:
        app: The Dash app instance
        options_analysis: The options analysis instance
        
    Returns:
        html.Div: The simplified dashboard layout
    """
    return html.Div([
        # Header with app title and symbol search
        dbc.Row([
            dbc.Col(html.H1("Options Recommendation Platform"), width=6),
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(id="symbol-input", type="text", value="AAPL", placeholder="Enter stock symbol"),
                    dbc.Button("Search", id="submit-button", color="primary"),
                ], size="lg"),
            ], width=6, className="d-flex align-items-center justify-content-end")
        ], className="header-row mb-3 mt-2"),
        
        # Authentication status
        dbc.Row([
            dbc.Col([
                html.Div(id="auth-status", children=[
                    html.Div("Checking authentication status...", id="auth-message"),
                    html.Div(id="auth-details", style={"display": "none"})
                ], className="auth-status-container")
            ])
        ], className="mb-3"),
        
        # Main content area with recommendations focus
        dbc.Row([
            # Left sidebar with timeframe filter and market overview
            dbc.Col([
                # Timeframe filter card
                dbc.Card([
                    dbc.CardHeader(html.H5("Trading Timeframe")),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id="trading-timeframe",
                            options=[
                                {"label": "15 Minutes", "value": "15min"},
                                {"label": "30 Minutes", "value": "30min"},
                                {"label": "60 Minutes", "value": "60min"},
                                {"label": "120 Minutes", "value": "120min"}
                            ],
                            value="30min",
                            inline=False,
                            className="mb-2"
                        ),
                        dbc.Button("Apply Timeframe", id="apply-timeframe", color="primary", size="sm", className="mt-2")
                    ])
                ], className="mb-3"),
                
                # Market overview card
                dbc.Card([
                    dbc.CardHeader(html.H5("Market Overview")),
                    dbc.CardBody([
                        html.Div(id="market-overview-content", children=[
                            # This will be populated with market data
                            html.P("Loading market data...", className="text-muted")
                        ])
                    ])
                ], className="mb-3"),
                
                # Watchlist card
                dbc.Card([
                    dbc.CardHeader(html.H5("Watchlist")),
                    dbc.CardBody([
                        html.Div(id="watchlist-content", children=[
                            # This will be populated with watchlist items
                            html.P("Your watchlist is empty", className="text-muted")
                        ])
                    ])
                ])
            ], width=3),
            
            # Main content area with recommendations
            dbc.Col([
                # Recommendations header with filters
                dbc.Row([
                    dbc.Col([
                        html.H3("Top Recommendations", className="mb-0")
                    ], width=6),
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button("All", id="filter-all", color="primary", outline=True, active=True, size="sm"),
                            dbc.Button("Calls", id="filter-calls", color="success", outline=True, size="sm"),
                            dbc.Button("Puts", id="filter-puts", color="danger", outline=True, size="sm")
                        ], className="mr-2"),
                        dbc.Button("Settings", id="recommendations-settings", color="secondary", outline=True, size="sm")
                    ], width=6, className="d-flex align-items-center justify-content-end")
                ], className="mb-3"),
                
                # Recommendations container
                html.Div([
                    dbc.Spinner(
                        html.Div(id="recommendations-container", className="recommendations-grid"),
                        color="primary",
                        type="border",
                        fullscreen=False
                    )
                ], className="mb-4"),
                
                # Validation visualization area
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Recommendation Validation", className="d-inline"),
                        dbc.Button(
                            "Expand",
                            id="expand-validation",
                            color="link",
                            size="sm",
                            className="float-right"
                        )
                    ]),
                    dbc.Collapse(
                        dbc.CardBody([
                            html.Div(id="validation-content", children=[
                                # This will be populated with validation visualizations
                                html.P("Select a recommendation to view validation details", className="text-muted text-center py-5")
                            ])
                        ]),
                        id="validation-collapse",
                        is_open=True
                    )
                ])
            ], width=9)
        ]),
        
        # Footer with access to other features
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Additional Features", className="mb-3"),
                        dbc.Tabs([
                            dbc.Tab(label="Options Chain", tab_id="tab-options-chain"),
                            dbc.Tab(label="Greeks", tab_id="tab-greeks"),
                            dbc.Tab(label="Technical Indicators", tab_id="tab-indicators"),
                            dbc.Tab(label="Historical Data", tab_id="tab-historical"),
                            dbc.Tab(label="Real-Time Data", tab_id="tab-realtime")
                        ], id="feature-tabs", active_tab="tab-options-chain"),
                        html.Div(id="feature-content", className="mt-3")
                    ])
                ])
            ])
        ], className="mt-4"),
        
        # Hidden divs for storing data
        html.Div(id="stored-recommendations", style={"display": "none"}),
        html.Div(id="selected-recommendation", style={"display": "none"}),
        html.Div(id="market-data-store", style={"display": "none"}),
        
        # Settings modal
        dbc.Modal([
            dbc.ModalHeader("Recommendation Settings"),
            dbc.ModalBody([
                # Confidence threshold
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
                
                # Risk-reward threshold
                html.Label("Minimum Risk-Reward Ratio:"),
                dbc.Input(
                    id="recommendation-risk-reward-threshold",
                    type="number",
                    min=1,
                    max=5,
                    step=0.5,
                    value=1.5,
                    className="mb-3"
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
                ], className="mb-2")
            ]),
            dbc.ModalFooter([
                dbc.Button("Reset to Defaults", id="reset-settings", color="secondary", className="mr-2"),
                dbc.Button("Apply", id="apply-settings", color="primary")
            ])
        ], id="settings-modal", size="lg", is_open=False),
        
        # Recommendation detail modal
        dbc.Modal([
            dbc.ModalHeader(html.H4(id="detail-modal-title")),
            dbc.ModalBody([
                html.Div(id="recommendation-detail-content")
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-detail-modal", color="secondary"),
                dbc.Button("Trade Now", id="trade-now-button", color="success")
            ])
        ], id="recommendation-detail-modal", size="xl", is_open=False)
    ])
