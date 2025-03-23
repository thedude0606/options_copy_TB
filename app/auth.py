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
                client.set_tokens(tokens)
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
            
            # Get the authorization URL
            auth_url = client.get_auth_url()
            
            # Open the browser
            webbrowser.open(auth_url)
            
            # Get the callback URL from the user
            callback_url = input("Paste the URL from your browser here: ")
            
            # Complete the authentication
            client.auth_callback(callback_url)
            
            # Save the tokens for future use
            with open(self.tokens_file, 'w') as f:
                json.dump(client.get_tokens(), f)
            
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
