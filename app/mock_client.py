"""
Mock client for development purposes.
Provides a mock implementation of the Schwab API client.
"""
import logging

class MockSchwabClient:
    """
    Mock implementation of the Schwab API client for development purposes.
    This allows the dashboard to run without requiring actual API credentials.
    """
    
    def __init__(self):
        """
        Initialize the mock client.
        """
        self.logger = logging.getLogger('mock_schwab_client')
        self.logger.info("Initializing MockSchwabClient")
        
    def get_quote(self, symbol):
        """
        Get a mock quote for the specified symbol.
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Mock quote data
        """
        self.logger.info(f"Mock get_quote for {symbol}")
        return {
            'symbol': symbol,
            'lastPrice': 150.0,
            'bidPrice': 149.5,
            'askPrice': 150.5,
            'volume': 1000000,
            'volatility': 0.25,
            'timestamp': '2025-03-27T12:00:00'
        }
    
    def get_option_chain(self, symbol, **kwargs):
        """
        Get a mock option chain for the specified symbol.
        
        Args:
            symbol (str): The stock symbol
            **kwargs: Additional parameters
            
        Returns:
            dict: Mock option chain data
        """
        self.logger.info(f"Mock get_option_chain for {symbol}")
        return {
            'symbol': symbol,
            'status': 'SUCCESS',
            'underlying': {
                'symbol': symbol,
                'description': f"{symbol} Inc.",
                'lastPrice': 150.0,
                'openPrice': 148.0,
                'highPrice': 152.0,
                'lowPrice': 147.0,
                'closePrice': 149.0,
                'volatility': 0.25,
                'volume': 1000000
            },
            'callExpDateMap': {},
            'putExpDateMap': {}
        }
    
    def get_price_history(self, symbol, **kwargs):
        """
        Get mock price history for the specified symbol.
        
        Args:
            symbol (str): The stock symbol
            **kwargs: Additional parameters
            
        Returns:
            dict: Mock price history data
        """
        self.logger.info(f"Mock get_price_history for {symbol}")
        return {
            'symbol': symbol,
            'candles': [
                {
                    'open': 148.0,
                    'high': 152.0,
                    'low': 147.0,
                    'close': 150.0,
                    'volume': 1000000,
                    'datetime': 1711584000000  # Example timestamp
                }
            ],
            'empty': False
        }
    
    def search_instruments(self, symbol, **kwargs):
        """
        Search for mock instruments matching the symbol.
        
        Args:
            symbol (str): The search symbol
            **kwargs: Additional parameters
            
        Returns:
            dict: Mock search results
        """
        self.logger.info(f"Mock search_instruments for {symbol}")
        return {
            symbol: {
                'symbol': symbol,
                'description': f"{symbol} Inc.",
                'exchange': 'NASDAQ',
                'assetType': 'EQUITY'
            }
        }
