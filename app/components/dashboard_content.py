"""
Dashboard content component for the options recommendation platform.
Provides the main content area for the dashboard.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

def create_dashboard_content():
    """
    Create the main dashboard content component
    
    Returns:
        html.Div: Dashboard content component with tabs and content areas
    """
    return html.Div([
        # Tabs for different dashboard views
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Overview",
                    tab_id="tab-overview",
                    children=create_overview_tab(),
                ),
                dbc.Tab(
                    label="Options Chain",
                    tab_id="tab-options-chain",
                    children=create_placeholder_tab("Options Chain"),
                ),
                dbc.Tab(
                    label="Greeks",
                    tab_id="tab-greeks",
                    children=create_placeholder_tab("Greeks Analysis"),
                ),
                dbc.Tab(
                    label="Technical Indicators",
                    tab_id="tab-indicators",
                    children=create_placeholder_tab("Technical Indicators"),
                ),
                dbc.Tab(
                    label="Historical Data",
                    tab_id="tab-historical",
                    children=create_placeholder_tab("Historical Data"),
                ),
                dbc.Tab(
                    label="Real-Time Data",
                    tab_id="tab-real-time",
                    children=create_placeholder_tab("Real-Time Data"),
                ),
            ],
            id="dashboard-tabs",
            active_tab="tab-overview",
        ),
        
        # Content area below tabs
        html.Div(id="tab-content", className="mt-4")
    ])

def create_overview_tab():
    """
    Create the overview tab content
    
    Returns:
        html.Div: Overview tab content
    """
    return html.Div([
        dbc.Row([
            # Market summary cards
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Market Summary"),
                    dbc.CardBody(
                        html.Div(id="market-overview-table")
                    )
                ], className="mb-4"),
                
                # Recent recommendations
                dbc.Card([
                    dbc.CardHeader("Recent Recommendations"),
                    dbc.CardBody(
                        html.Div(id="recent-recommendations")
                    )
                ])
            ], width=4),
            
            # Main chart area
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col(html.H5("Price Chart"), width="auto"),
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.Input(id="chart-symbol-input", placeholder="Enter symbol", value="SPY"),
                                    dbc.Select(
                                        id="chart-timeframe-select",
                                        options=[
                                            {"label": "1D", "value": "1d"},
                                            {"label": "5D", "value": "5d"},
                                            {"label": "1M", "value": "1m"},
                                            {"label": "3M", "value": "3m"},
                                            {"label": "6M", "value": "6m"},
                                            {"label": "1Y", "value": "1y"},
                                            {"label": "YTD", "value": "ytd"},
                                        ],
                                        value="1m",
                                    ),
                                    dbc.Button("Update", id="update-chart-button", color="primary"),
                                ], size="sm"),
                            ], width="auto", className="ms-auto"),
                        ], align="center"),
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-price-chart",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="price-chart",
                                    figure=create_empty_chart("Enter a symbol and click Update"),
                                    style={"height": "400px"},
                                )
                            ]
                        )
                    ])
                ], className="mb-4"),
                
                # Validation visualization
                dbc.Card([
                    dbc.CardHeader("Recommendation Validation"),
                    dbc.CardBody([
                        html.Div(id="validation-content")
                    ])
                ])
            ], width=8)
        ])
    ])

def create_placeholder_tab(title):
    """
    Create a placeholder tab content
    
    Args:
        title (str): Tab title
        
    Returns:
        html.Div: Placeholder tab content
    """
    return html.Div([
        dbc.Alert(
            f"The {title} tab content will be loaded when selected.",
            color="info",
            className="mb-4"
        ),
        html.Div(id=f"{title.lower().replace(' ', '-')}-content")
    ])

def create_empty_chart(message):
    """
    Create an empty chart with a message
    
    Args:
        message (str): Message to display
        
    Returns:
        go.Figure: Empty chart figure
    """
    fig = go.Figure()
    
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 16
                }
            }
        ],
        template="plotly_white"
    )
    
    return fig
