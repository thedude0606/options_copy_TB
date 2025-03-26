"""
Main entry point for the simplified options recommendation platform.
Implements a Robinhood-inspired UI focused on short-term options trading.
"""
import os
import sys
import logging
from datetime import datetime

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Import custom modules
from app.simplified_layout import create_simplified_layout
from app.data_collector import DataCollector
from app.integration import initialize_components, register_callbacks, run_all_tests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("options_platform.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('main')

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Options Recommendation Platform"

# Initialize data collector
try:
    logger.info("Initializing data collector...")
    data_collector = DataCollector()
    logger.info("Data collector initialized successfully")
except Exception as e:
    logger.error(f"Error initializing data collector: {str(e)}")
    data_collector = None

# Initialize components
try:
    if data_collector:
        logger.info("Initializing platform components...")
        data_pipeline, recommendation_engine = initialize_components(app, data_collector)
        logger.info("Platform components initialized successfully")
    else:
        logger.error("Cannot initialize components: data collector is None")
        data_pipeline, recommendation_engine = None, None
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    data_pipeline, recommendation_engine = None, None

# Create app layout
try:
    logger.info("Creating app layout...")
    app.layout = create_simplified_layout()
    logger.info("App layout created successfully")
except Exception as e:
    logger.error(f"Error creating app layout: {str(e)}")
    app.layout = html.Div("Error initializing application. Please check logs.")

# Register callbacks
try:
    if data_pipeline and recommendation_engine:
        logger.info("Registering callbacks...")
        register_callbacks(app, data_pipeline, recommendation_engine)
        logger.info("Callbacks registered successfully")
    else:
        logger.error("Cannot register callbacks: components are None")
except Exception as e:
    logger.error(f"Error registering callbacks: {str(e)}")

# Run tests in development mode
if os.environ.get('ENVIRONMENT') == 'development':
    try:
        if data_pipeline and recommendation_engine:
            logger.info("Running tests in development mode...")
            test_results = run_all_tests(recommendation_engine, data_pipeline)
            
            # Log test results
            success_rate = test_results.get('summary', {}).get('overall_success_rate', 0)
            logger.info(f"Test results: {success_rate:.2%} success rate")
            
            if success_rate < 0.8:
                logger.warning("Test success rate below 80%. Some features may not work correctly.")
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")

# Main entry point
if __name__ == '__main__':
    logger.info("Starting application...")
    app.run_server(debug=True, host='0.0.0.0')
