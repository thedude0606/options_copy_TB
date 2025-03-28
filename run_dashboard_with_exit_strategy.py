"""
Main dashboard application for options trading with exit strategy prediction.
Integrates enhanced machine learning, risk management, and exit strategy features.
"""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import webbrowser
import os
import sys
import logging
from datetime import datetime, timedelta

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from app.components.exit_strategy_display import create_exit_strategy_tab, register_exit_strategy_callbacks, ExitStrategyDisplay
from app.components.recommendations_tab import create_recommendations_tab, register_recommendations_callbacks

# Import data collectors and API clients
from app.options_data import OptionsDataRetriever
from app.data_collector import DataCollector
from app.data.options_collector import OptionsDataCollector  # Added missing import
from schwabdev.client import Client as SchwabClient

# Import recommendation engine with exit strategy
from app.analysis.exit_strategy_recommendation_engine import ExitStrategyEnhancedRecommendationEngine

# Import database module (mock implementation for standalone usage)
class MockDatabase:
    def __init__(self):
        self.logger = logging.getLogger('mock_database')
        self.logger.info("Initialized mock database")
    
    def store_options_data(self, options_data):
        self.logger.info(f"Mock storing {len(options_data)} options data records")
        return True
    
    def store_underlying_data(self, underlying_data):
        self.logger.info(f"Mock storing {len(underlying_data)} underlying data records")
        return True
    
    def get_options_data(self, symbol=None, start_date=None, end_date=None):
        self.logger.info(f"Mock retrieving options data for {symbol}")
        return []
    
    def get_underlying_data(self, symbol=None, start_date=None, end_date=None):
        self.logger.info(f"Mock retrieving underlying data for {symbol}")
        return []

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dashboard_exit_strategy.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('dashboard_exit_strategy')
    logger.info("Starting Options Dashboard with Exit Strategy Prediction")
    
    # Initialize the Schwab API client with authentication from environment variables
    app_key = os.getenv('app_key')
    app_secret = os.getenv('app_secret')
    callback_url = os.getenv('callback_url', 'https://127.0.0.1')
    
    if not app_key or not app_secret:
        logger.warning("API credentials not found in environment variables. Using default values for development.")
        app_key = "YOUR_APP_KEY"  # Replace with your actual key when deploying
        app_secret = "YOUR_APP_SECRET"  # Replace with your actual secret when deploying
    
    # Initialize API client
    client = SchwabClient(app_key, app_secret, callback_url)
    logger.info("Schwab API client initialized with authentication")
    
    # Initialize mock database for standalone usage
    db = MockDatabase()
    
    # Initialize the options data collector with required parameters
    data_collector = OptionsDataCollector(api_client=client, db=db)
    logger.info("Options data collector initialized")
    
    # Initialize the data collector for technical indicators
    technical_data_collector = DataCollector()
    
    # Initialize recommendation engine with exit strategy prediction
    recommendation_engine = ExitStrategyEnhancedRecommendationEngine(
        data_collector=technical_data_collector,
        ml_config_path='config/ml_config.json',
        debug=True
    )
    logger.info("Exit Strategy Enhanced Recommendation Engine initialized")
    
    # Initialize Dash app
    app = dash.Dash(
        __name__, 
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )
    
    # Define app layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Options Trading Platform with Exit Strategy Prediction", className="display-4"),
            html.P("Complete options trading recommendations with entry and exit strategies", className="lead"),
            html.Hr()
        ], className="container mt-4"),
        
        # Main content
        html.Div([
            # Tabs
            dcc.Tabs([
                dcc.Tab(label="Recommendations", value="recommendations-tab", children=create_recommendations_tab()),
                dcc.Tab(label="Exit Strategies", value="exit-strategies-tab", children=create_exit_strategy_tab()),
                # Other tabs can be added here
            ]),
        ], className="container"),
        
        # Interval component for periodic updates
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # in milliseconds (1 minute)
            n_intervals=0
        )
    ])
    
    # Register callbacks
    register_exit_strategy_callbacks(app, recommendation_engine)
    register_recommendations_callbacks(app, recommendation_engine)
    
    # Callback for recommendations tab
    @app.callback(
        dash.Output("recommendations-output", "children"),
        [dash.Input("interval-component", "n_intervals")]
    )
    def update_recommendations(n_intervals):
        # Generate recommendations with exit strategies
        try:
            recommendations = recommendation_engine.generate_recommendations(
                symbol='AAPL',  # Example symbol
                lookback_days=30,
                confidence_threshold=0.6,
                max_recommendations=5
            )
            
            # If no recommendations, return message
            if recommendations.empty:
                return html.Div("No recommendations available at this time.")
            
            # Convert to list of dictionaries
            rec_list = recommendations.to_dict('records')
            
            # Create cards for each recommendation
            cards = []
            for rec in rec_list:
                # Create recommendation card
                rec_card = dbc.Card(
                    dbc.CardBody([
                        html.H5(f"{rec.get('underlying', '')} {rec.get('option_type', '')} {rec.get('strike', 0)}", className="card-title"),
                        html.H6(f"Expiration: {rec.get('expiration_date', '')}", className="card-subtitle mb-2 text-muted"),
                        
                        html.Div([
                            html.Strong("Entry Price:"),
                            html.Span(f" ${rec.get('price', 0):.2f}", className="ms-2")
                        ], className="mb-2"),
                        
                        html.Div([
                            html.Strong("Confidence:"),
                            html.Span(f" {rec.get('confidence', 0) * 100:.1f}%", className="ms-2")
                        ], className="mb-2"),
                        
                        html.Hr(),
                        
                        # Add exit strategy display
                        ExitStrategyDisplay.create_exit_strategy_card(rec)
                    ])
                )
                cards.append(dbc.Col(rec_card, width=6, className="mb-4"))
            
            # Arrange cards in rows
            rows = []
            for i in range(0, len(cards), 2):
                row_cards = cards[i:i+2]
                rows.append(dbc.Row(row_cards, className="mb-4"))
            
            return html.Div(rows)
            
        except Exception as e:
            return html.Div(f"Error generating recommendations: {str(e)}")
    
    # Run the app
    app.run_server(debug=True, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()
