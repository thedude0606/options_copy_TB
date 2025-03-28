"""
Test script for the updated get_price_data method with lookback_days parameter.
This script bypasses the token validation to test the functionality.
"""

import sys
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('lookback_days_test')

# Monkey patch the Tokens class to bypass validation
import schwabdev.tokens
original_init = schwabdev.tokens.Tokens.__init__

def patched_init(self, client, app_key, app_secret, callback_url, tokens_file="tokens.json", capture_callback=False, call_on_notify=None):
    """Patched init method to bypass token validation"""
    # Skip validation checks
    self._client = client
    self._app_key = app_key
    self._app_secret = app_secret
    self._callback_url = callback_url
    self._tokens_file = tokens_file
    self._capture_callback = capture_callback
    self.call_on_notify = call_on_notify
    
    # Set mock tokens
    self.access_token = "mock_access_token"
    self.refresh_token = "mock_refresh_token"
    self.id_token = "mock_id_token"
    
    # Set other attributes
    import datetime
    self._access_token_issued = datetime.datetime.now(datetime.timezone.utc)
    self._refresh_token_issued = datetime.datetime.now(datetime.timezone.utc)
    self._access_token_timeout = 1800
    self._refresh_token_timeout = 7 * 24 * 60 * 60
    
    # Log initialization
    client.logger.info("Mock tokens initialized for testing")

# Apply the monkey patch
schwabdev.tokens.Tokens.__init__ = patched_init

# Create a mock get_client function to bypass authentication
def mock_get_client(interactive=False):
    from schwabdev.client import Client
    return Client("test_key", "test_secret", "https://example.com")

# Patch the auth module
import sys
import types
mock_auth_module = types.ModuleType('app.auth')
mock_auth_module.get_client = mock_get_client
sys.modules['app.auth'] = mock_auth_module

# Now import and test the DataCollector
from app.data_collector import DataCollector

def test_lookback_days_parameter():
    """Test the lookback_days parameter in get_price_data method"""
    logger.info("Creating DataCollector...")
    data_collector = DataCollector(interactive_auth=False)
    
    # Test with lookback_days parameter
    logger.info("Testing get_price_data with lookback_days=30...")
    data_with_lookback = data_collector.get_price_data('AAPL', lookback_days=30)
    logger.info(f"Shape with lookback_days=30: {data_with_lookback.shape}")
    
    # Test with period parameter
    logger.info("Testing get_price_data with period=10...")
    data_with_period = data_collector.get_price_data('AAPL', period=10)
    logger.info(f"Shape with period=10: {data_with_period.shape}")
    
    # Verify that lookback_days produces more data than period
    logger.info(f"Rows with lookback_days=30: {len(data_with_lookback)}")
    logger.info(f"Rows with period=10: {len(data_with_period)}")
    
    # Test that lookback_days is properly used
    assert len(data_with_lookback) > len(data_with_period), "lookback_days should produce more data than period"
    
    # Test that the implementation works with the exit_strategy_predictor
    logger.info("Testing compatibility with exit_strategy_predictor usage...")
    option_data = {
        'symbol': 'AAPL_C150',
        'underlying': 'AAPL',
        'option_type': 'CALL',
        'strike': 150.0,
        'expiration_date': '2025-04-18'
    }
    
    # Simulate the call from exit_strategy_predictor
    try:
        price_data = data_collector.get_price_data(option_data['underlying'], lookback_days=30)
        logger.info(f"Successfully retrieved price data with lookback_days from exit_strategy_predictor simulation")
        logger.info(f"Price data shape: {price_data.shape}")
        return True
    except Exception as e:
        logger.error(f"Error in exit_strategy_predictor simulation: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting lookback_days parameter test...")
    success = test_lookback_days_parameter()
    logger.info(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
