import os
import json
import webbrowser
from dotenv import load_dotenv
from schwabdev.client import Client

# Load environment variables from .env file
load_dotenv()

# Flag to enable mock mode for testing without actual authentication
MOCK_MODE = False

class MockClient:
    """
    Mock client that mimics the Schwab API client for testing purposes
    """
    def __init__(self, app_key=None, app_secret=None, callback_url=None):
        self.app_key = app_key
        self.app_secret = app_secret
        self.callback_url = callback_url
        
    def quotes(self, symbols=None):
        """Mock implementation of quotes method"""
        mock_response = {
            "symbol": symbols[0] if symbols else "SPY",
            "description": "SPDR S&P 500 ETF Trust",
            "bidPrice": 450.25,
            "askPrice": 450.45,
            "lastPrice": 450.35,
            "openPrice": 448.70,
            "highPrice": 452.10,
            "lowPrice": 447.90,
            "closePrice": 450.35,
            "totalVolume": 75000000,
            "quoteTimeInLong": 1616432400000,
            "tradeTimeInLong": 1616432400000,
            "mark": 450.35,
            "exchange": "NYSE",
            "exchangeName": "New York Stock Exchange",
            "volatility": 0.0123,
            "52WkHigh": 455.0,
            "52WkLow": 380.0,
            "peRatio": 22.5,
            "divAmount": 5.65,
            "divYield": 1.25,
            "divDate": "2023-03-15",
            "regularMarketLastPrice": 450.35,
            "regularMarketLastSize": 100,
            "regularMarketNetChange": 1.65,
            "regularMarketTradeTimeInLong": 1616432400000,
            "netPercentChangeInDouble": 0.37,
            "markChangeInDouble": 1.65,
            "markPercentChangeInDouble": 0.37,
            "regularMarketPercentChangeInDouble": 0.37
        }
        
        class MockResponse:
            def __init__(self, data):
                self.data = data
                self.status_code = 200
                
            def json(self):
                return [self.data]
                
        return MockResponse(mock_response)
    
    def option_chains(self, symbol=None, contractType=None, strikeCount=None, includeUnderlyingQuote=None, strategy=None):
        """Mock implementation of option_chains method"""
        mock_response = {
            "symbol": symbol,
            "status": "SUCCESS",
            "underlying": {
                "symbol": symbol,
                "description": f"{symbol} Stock",
                "change": 1.65,
                "percentChange": 0.37,
                "close": 448.70,
                "quoteTime": 1616432400000,
                "tradeTime": 1616432400000,
                "bid": 450.25,
                "ask": 450.45,
                "last": 450.35,
                "mark": 450.35,
                "markChange": 1.65,
                "markPercentChange": 0.37,
                "bidSize": 100,
                "askSize": 100,
                "highPrice": 452.10,
                "lowPrice": 447.90,
                "openPrice": 448.70,
                "totalVolume": 75000000,
                "exchangeName": "NYSE",
                "fiftyTwoWeekHigh": 455.0,
                "fiftyTwoWeekLow": 380.0,
                "delayed": False
            },
            "callExpDateMap": {
                "2023-04-21:30": {
                    "450.0": [
                        {
                            "putCall": "CALL",
                            "symbol": f"{symbol}_042123C450",
                            "description": f"{symbol} Apr 21 2023 450 Call",
                            "exchangeName": "OPR",
                            "bid": 5.25,
                            "ask": 5.45,
                            "last": 5.35,
                            "mark": 5.35,
                            "bidSize": 10,
                            "askSize": 10,
                            "bidAskSize": "10X10",
                            "lastSize": 0,
                            "highPrice": 5.50,
                            "lowPrice": 5.10,
                            "openPrice": 5.20,
                            "closePrice": 5.30,
                            "totalVolume": 1500,
                            "tradeDate": None,
                            "tradeTimeInLong": 1616432400000,
                            "quoteTimeInLong": 1616432400000,
                            "netChange": 0.05,
                            "volatility": 25.0,
                            "delta": 0.55,
                            "gamma": 0.05,
                            "theta": -0.15,
                            "vega": 0.10,
                            "rho": 0.05,
                            "openInterest": 5000,
                            "timeValue": 5.35,
                            "theoreticalOptionValue": 5.35,
                            "theoreticalVolatility": 25.0,
                            "strikePrice": 450.0,
                            "expirationDate": 1682035200000,
                            "daysToExpiration": 30,
                            "expirationType": "R",
                            "lastTradingDay": 1682035200000,
                            "multiplier": 100.0,
                            "settlementType": " ",
                            "deliverableNote": "",
                            "isIndexOption": None,
                            "percentChange": 0.94,
                            "markChange": 0.05,
                            "markPercentChange": 0.94,
                            "intrinsicValue": 0.35,
                            "inTheMoney": True,
                            "mini": False,
                            "nonStandard": False
                        }
                    ]
                }
            },
            "putExpDateMap": {
                "2023-04-21:30": {
                    "450.0": [
                        {
                            "putCall": "PUT",
                            "symbol": f"{symbol}_042123P450",
                            "description": f"{symbol} Apr 21 2023 450 Put",
                            "exchangeName": "OPR",
                            "bid": 4.75,
                            "ask": 4.95,
                            "last": 4.85,
                            "mark": 4.85,
                            "bidSize": 10,
                            "askSize": 10,
                            "bidAskSize": "10X10",
                            "lastSize": 0,
                            "highPrice": 5.00,
                            "lowPrice": 4.60,
                            "openPrice": 4.70,
                            "closePrice": 4.80,
                            "totalVolume": 1200,
                            "tradeDate": None,
                            "tradeTimeInLong": 1616432400000,
                            "quoteTimeInLong": 1616432400000,
                            "netChange": 0.05,
                            "volatility": 25.0,
                            "delta": -0.45,
                            "gamma": 0.05,
                            "theta": -0.15,
                            "vega": 0.10,
                            "rho": -0.05,
                            "openInterest": 4500,
                            "timeValue": 4.85,
                            "theoreticalOptionValue": 4.85,
                            "theoreticalVolatility": 25.0,
                            "strikePrice": 450.0,
                            "expirationDate": 1682035200000,
                            "daysToExpiration": 30,
                            "expirationType": "R",
                            "lastTradingDay": 1682035200000,
                            "multiplier": 100.0,
                            "settlementType": " ",
                            "deliverableNote": "",
                            "isIndexOption": None,
                            "percentChange": 1.04,
                            "markChange": 0.05,
                            "markPercentChange": 1.04,
                            "intrinsicValue": 0.0,
                            "inTheMoney": False,
                            "mini": False,
                            "nonStandard": False
                        }
                    ]
                }
            }
        }
        
        class MockResponse:
            def __init__(self, data):
                self.data = data
                self.status_code = 200
                
            def json(self):
                return self.data
                
        return MockResponse(mock_response)

class SchwabAuth:
    """
    Class to handle Schwab API authentication
    """
    def __init__(self):
        # Get credentials from environment variables
        self.app_key = os.getenv('SCHWAB_APP_KEY', 'mock_app_key')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET', 'mock_app_secret')
        self.callback_url = os.getenv('SCHWAB_CALLBACK_URL', 'https://127.0.0.1:8080')
        self.tokens_file = 'tokens.json'
        
        # In mock mode, we don't validate credentials
        if not MOCK_MODE and not all([self.app_key, self.app_secret, self.callback_url]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
    
    def get_client(self, interactive=False):
        """
        Create and return an authenticated Schwab API client
        
        Args:
            interactive (bool): Whether to allow interactive authentication
            
        Returns:
            Client: Authenticated Schwab API client or MockClient in mock mode
        """
        # If in mock mode, return a mock client
        if MOCK_MODE:
            return MockClient(
                app_key=self.app_key,
                app_secret=self.app_secret,
                callback_url=self.callback_url
            )
            
        # Create client
        client = Client(
            app_key=self.app_key,
            app_secret=self.app_secret,
            callback_url=self.callback_url
        )
        
        # If tokens exist, try to use them
        if os.path.exists(self.tokens_file):
            try:
                with open(self.tokens_file, 'r') as f:
                    tokens = json.load(f)
                # The tokens object structure is different in the current Schwabdev version
                # We need to adapt to the expected structure
                if "token_dictionary" in tokens:
                    token_dict = tokens["token_dictionary"]
                    client.tokens.access_token = token_dict.get("access_token")
                    client.tokens.refresh_token = token_dict.get("refresh_token")
                    client.tokens.id_token = token_dict.get("id_token")
                else:
                    # Try direct assignment if token_dictionary is not present
                    client.tokens.access_token = tokens.get("access_token")
                    client.tokens.refresh_token = tokens.get("refresh_token")
                    client.tokens.id_token = tokens.get("id_token")
                return client
            except Exception as e:
                print(f"Error loading tokens: {str(e)}")
                # If interactive mode is enabled, continue to interactive auth
                if not interactive:
                    raise
        
        # If interactive mode is enabled, perform interactive authentication
        if interactive:
            print("Authentication required. Please follow these steps:")
            print("1. A browser window will open with the Schwab authentication page")
            print("2. Log in with your Schwab credentials and authorize the application")
            print("3. You will be redirected to a page that may show an error (this is expected)")
            print("4. Copy the full URL from your browser's address bar and paste it below")
            
            # Generate the authorization URL directly instead of using client.get_auth_url()
            auth_url = f'https://api.schwabapi.com/v1/oauth/authorize?client_id={self.app_key}&redirect_uri={self.callback_url}'
            
            # Open the browser
            webbrowser.open(auth_url)
            
            # Get the callback URL from the user
            callback_url = input("Paste the URL from your browser here: ")
            
            # Use client.tokens._update_refresh_token_from_code instead of client.auth_callback
            # Extract the code from the callback URL
            try:
                code = callback_url.split("code=")[1].split("&")[0]
                client.tokens._update_refresh_token_from_code(code)
            except Exception as e:
                print(f"Error processing callback URL: {str(e)}")
                raise
            
            return client
        else:
            raise ValueError("Authentication required but interactive mode is disabled")

# Singleton instance for easy access
auth_manager = SchwabAuth()

def get_client(interactive=False):
    """
    Get an authenticated Schwab API client
    
    Args:
        interactive (bool): Whether to allow interactive authentication
        
    Returns:
        Client: Authenticated Schwab API client or MockClient in mock mode
    """
    return auth_manager.get_client(interactive=interactive)
