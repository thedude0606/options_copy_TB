"""
Test script for the implemented API methods.
This script bypasses the token validation to test the functionality.
"""

import sys
import logging
from schwabdev.client import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('api_test')

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

def test_client_methods():
    """Test the implemented client methods"""
    logger.info("Creating test client...")
    client = Client("test_key", "test_secret", "https://example.com")
    
    # Test get_user_principals
    logger.info("Testing get_user_principals...")
    user_principals = client.get_user_principals()
    logger.info(f"get_user_principals returned: {user_principals is not None}")
    if user_principals:
        logger.info(f"User ID: {user_principals.get('userId')}")
    
    # Test get_options_chain
    logger.info("Testing get_options_chain...")
    options_chain = client.get_options_chain("AAPL")
    logger.info(f"get_options_chain returned: {options_chain is not None}")
    if options_chain:
        logger.info(f"Options chain keys: {list(options_chain.keys())}")
        logger.info(f"Number of call expirations: {len(options_chain.get('callExpDateMap', {}))}")
        logger.info(f"Number of put expirations: {len(options_chain.get('putExpDateMap', {}))}")
    
    # Test get_quote
    logger.info("Testing get_quote...")
    quote = client.get_quote("AAPL")
    logger.info(f"get_quote returned: {quote is not None}")
    if quote:
        logger.info(f"Quote data: {quote}")
    
    return all([user_principals, options_chain, quote])

def test_data_collector():
    """Test the DataCollector class with the implemented get_price_data method"""
    from app.data_collector import DataCollector
    
    logger.info("Creating DataCollector...")
    data_collector = DataCollector(interactive_auth=False)
    
    # Test get_price_data
    logger.info("Testing get_price_data...")
    price_data = data_collector.get_price_data("AAPL")
    logger.info(f"get_price_data returned DataFrame with shape: {price_data.shape if hasattr(price_data, 'shape') else None}")
    if not price_data.empty:
        logger.info(f"Price data columns: {price_data.columns.tolist()}")
        logger.info(f"Price data sample:\n{price_data.head(3)}")
    
    # Test get_technical_indicators
    logger.info("Testing get_technical_indicators...")
    indicators = data_collector.get_technical_indicators("AAPL")
    logger.info(f"get_technical_indicators returned DataFrame with shape: {indicators.shape if hasattr(indicators, 'shape') else None}")
    if not indicators.empty:
        logger.info(f"Technical indicators columns: {indicators.columns.tolist()}")
    
    return not price_data.empty and not indicators.empty

if __name__ == "__main__":
    logger.info("Starting API tests...")
    
    client_success = test_client_methods()
    logger.info(f"Client methods test {'PASSED' if client_success else 'FAILED'}")
    
    collector_success = test_data_collector()
    logger.info(f"DataCollector test {'PASSED' if collector_success else 'FAILED'}")
    
    overall_success = client_success and collector_success
    logger.info(f"Overall test result: {'PASSED' if overall_success else 'FAILED'}")
    
    sys.exit(0 if overall_success else 1)
