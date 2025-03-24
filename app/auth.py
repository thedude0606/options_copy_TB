import os
import json
import webbrowser
from dotenv import load_dotenv
from schwabdev.client import Client

# Load environment variables from .env file
load_dotenv()

class SchwabAuth:
    """
    Class to handle Schwab API authentication
    """
    def __init__(self):
        # Get credentials from environment variables
        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET')
        self.callback_url = os.getenv('SCHWAB_CALLBACK_URL')
        self.tokens_file = 'tokens.json'
        
        # Validate credentials
        if not all([self.app_key, self.app_secret, self.callback_url]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
    
    def get_client(self, interactive=False):
        """
        Create and return an authenticated Schwab API client
        
        Args:
            interactive (bool): Whether to allow interactive authentication
            
        Returns:
            Client: Authenticated Schwab API client
        """
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
        Client: Authenticated Schwab API client
    """
    return auth_manager.get_client(interactive=interactive)
