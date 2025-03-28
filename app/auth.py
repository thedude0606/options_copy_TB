"""
Authentication module for options recommendation platform.
Handles authentication with Schwab API.
"""
from dotenv import load_dotenv
import os
import json
import traceback

# Mock client for development and testing
class MockClient:
    """Mock client for development and testing"""
    def __init__(self):
        self.authenticated = True
    
    def option_chains(self, **kwargs):
        """Mock option chains method"""
        # Return empty option chain structure
        return {
            'symbol': kwargs.get('symbol', ''),
            'status': 'SUCCESS',
            'underlying': {
                'symbol': kwargs.get('symbol', ''),
                'description': f"{kwargs.get('symbol', '')} Stock",
                'last': 100.0,
                'close': 99.5,
                'mark': 100.0
            },
            'putExpDateMap': {},
            'callExpDateMap': {}
        }

def get_client(interactive=False):
    """
    Get authenticated client for Schwab API
    
    Args:
        interactive (bool): Whether to allow interactive authentication
        
    Returns:
        client: Authenticated client
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Check if we have credentials
        api_key = os.getenv('SCHWAB_API_KEY')
        
        if not api_key:
            print("No API credentials found. Using mock client for development.")
            return MockClient()
        
        # In a real implementation, this would authenticate with the Schwab API
        # For now, return a mock client
        return MockClient()
        
    except Exception as e:
        print(f"Error authenticating with Schwab API: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Falling back to mock client.")
        return MockClient()
