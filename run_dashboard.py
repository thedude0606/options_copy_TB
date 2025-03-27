"""
Main dashboard application for options trading.
Integrates enhanced machine learning and risk management features.
"""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import webbrowser
import os
import logging

# Import components
from app.components.indicators_tab import create_indicators_tab, register_indicators_callbacks
from app.components.greeks_tab import register_greeks_callbacks
from app.components.recommendations_tab import create_recommendations_tab, register_recommendations_callbacks

# Import data collectors and API clients
from app.options_data import OptionsDataRetriever
from app.data_collector import DataCollector
from schwabdev.client import Client as SchwabClient

# Import analysis modules
from app.analysis.enhanced_recommendation_engine import EnhancedRecommendationEngine

# Import debugging utilities
from app.utils.debug_utils import add_debug_div_to_layout, add_debug_callback_to_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('dashboard')
logger.info("Starting Options Dashboard with Enhanced ML Features")

# Initialize the Schwab API client with authentication from environment variables
app_key = os.getenv('app_key')
app_secret = os.getenv('app_secret')
callback_url = os.getenv('callback_url', 'https://127.0.0.1')

if not app_key or not app_secret:
    logger.warning("API credentials not found in environment variables. Using default values for development.")
    app_key = "YOUR_APP_KEY"  # Replace with your actual key when deploying
    app_secret = "YOUR_APP_SECRET"  # Replace with your actual secret when deploying

client = SchwabClient(app_key, app_secret, callback_url)
logger.info("Schwab API client initialized with authentication")

# Initialize the options data retriever
options_data_retriever = OptionsDataRetriever(client)

# Initialize the data collector for technical indicators
data_collector = DataCollector()

# Initialize the enhanced recommendation engine with ML capabilities
recommendation_engine = EnhancedRecommendationEngine(data_collector, debug=True)
logger.info("Enhanced recommendation engine initialized")

# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Add debugging callback for timeline selector
add_debug_callback_to_app(app)

# Create the app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Schwab Options Dashboard", className="display-4"),
        html.P("Real-time options data and analysis powered by Schwab API and ML", className="lead"),
        html.Hr()
    ], className="container mt-4"),
    
    # Main content
    html.Div([
        # Tabs
        dcc.Tabs(id="tabs", value="recommendations-tab", children=[
            dcc.Tab(label="Recommendations", value="recommendations-tab"),
            dcc.Tab(label="Technical Indicators", value="indicators-tab"),
            dcc.Tab(label="Greeks Analysis", value="greeks-tab")
        ]),
        
        # Tab content
        html.Div(id="tab-content", className="p-4")
    ], className="container"),
    
    # Store components for sharing data between callbacks
    dcc.Store(id="options-data-store"),
    dcc.Store(id="selected-options-store"),
    
    # Interval component for periodic updates
    dcc.Interval(
        id="interval-component",
        interval=60 * 1000,  # Update every 60 seconds
        n_intervals=0
    )
])

# Callback to render tab content
@app.callback(
    dash.Output("tab-content", "children"),
    dash.Input("tabs", "value")
)
def render_tab_content(tab):
    """
    Render the content for the selected tab
    
    Args:
        tab (str): Selected tab value
        
    Returns:
        html.Div: Tab content
    """
    if tab == "recommendations-tab":
        return create_recommendations_tab()
    elif tab == "indicators-tab":
        # Add the debug div to the indicators tab
        indicators_tab = create_indicators_tab()
        debug_div = add_debug_div_to_layout()
        
        # Combine the indicators tab with the debug div
        if isinstance(indicators_tab, html.Div) and hasattr(indicators_tab, 'children'):
            if isinstance(indicators_tab.children, list):
                indicators_tab.children.append(debug_div)
            else:
                indicators_tab.children = [indicators_tab.children, debug_div]
        
        return indicators_tab
    elif tab == "greeks-tab":
        return html.Div([
            html.H3("Options Greeks Analysis"),
            html.Div([
                html.Label("Symbol:"),
                dbc.InputGroup([
                    dbc.Input(id="greeks-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
                    dbc.Button("Analyze", id="greeks-analyze-button", color="primary")
                ], className="mb-3")
            ]),
            html.Div(id="greeks-content")
        ])
    else:
        return html.Div([
            html.H3("Tab content not implemented")
        ])

# Register callbacks
register_indicators_callbacks(app)
register_greeks_callbacks(app)
register_recommendations_callbacks(app, recommendation_engine)

# Run the app
if __name__ == "__main__":
    # Open browser automatically
    webbrowser.open_new("http://127.0.0.1:8050/")
    
    # Run the app
    app.run_server(debug=True, use_reloader=False)
