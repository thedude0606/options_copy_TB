"""
Schwab API client configuration for real data access.
This module ensures proper authentication and configuration for the Schwab API client.
"""

import os
import logging
import json
from dotenv import load_dotenv
from schwabdev.client import Client as SchwabClient

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger('schwab_api_config')

class SchwabAPIConfig:
    """
    Configuration manager for Schwab API client.
    Handles authentication, token management, and client initialization.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the Schwab API configuration.
        
        Args:
            config_file (str, optional): Path to configuration file with API credentials.
                If not provided, will look for environment variables.
        """
        self.logger = logger
        self.config_file = config_file
        self.app_key = None
        self.app_secret = None
        self.callback_url = None
        self.client = None
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load API credentials from config file or environment variables."""
        # Try loading from config file first
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.app_key = config.get('app_key')
                    self.app_secret = config.get('app_secret')
                    self.callback_url = config.get('callback_url', 'https://127.0.0.1')
                    self.logger.info(f"Loaded API credentials from config file: {self.config_file}")
            except Exception as e:
                self.logger.error(f"Error loading config file: {str(e)}")
        
        # If not loaded from file, try environment variables
        if not self.app_key or not self.app_secret:
            # Try with SCHWAB_ prefix (from .env file)
            self.app_key = os.getenv('SCHWAB_APP_KEY')
            self.app_secret = os.getenv('SCHWAB_APP_SECRET')
            self.callback_url = os.getenv('SCHWAB_CALLBACK_URL', 'https://127.0.0.1')
            
            # If not found, try legacy variable names
            if not self.app_key or not self.app_secret:
                self.app_key = os.getenv('app_key')
                self.app_secret = os.getenv('app_secret')
                self.callback_url = os.getenv('callback_url', self.callback_url)
            
            if self.app_key and self.app_secret:
                self.logger.info("Loaded API credentials from environment variables")
        
        # If still not loaded, use default values for development (will need to be replaced)
        if not self.app_key or not self.app_secret:
            self.logger.warning("API credentials not found. Using default values for development.")
            self.app_key = "YOUR_APP_KEY"  # Replace with actual key
            self.app_secret = "YOUR_APP_SECRET"  # Replace with actual secret
    
    def get_client(self):
        """
        Get an initialized Schwab API client with proper authentication.
        
        Returns:
            SchwabClient: Initialized and authenticated Schwab API client
        """
        if not self.client:
            self.client = SchwabClient(
                self.app_key, 
                self.app_secret, 
                self.callback_url
            )
            self.logger.info("Initialized Schwab API client with authentication")
        
        return self.client
    
    def save_configuration(self, app_key, app_secret, callback_url=None):
        """
        Save API credentials to configuration file.
        
        Args:
            app_key (str): Schwab API app key
            app_secret (str): Schwab API app secret
            callback_url (str, optional): Callback URL for OAuth
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config_file:
            self.config_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'schwab_api_config.json'
            )
        
        try:
            config = {
                'app_key': app_key,
                'app_secret': app_secret,
                'callback_url': callback_url or 'https://127.0.0.1'
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Update current values
            self.app_key = app_key
            self.app_secret = app_secret
            self.callback_url = callback_url or 'https://127.0.0.1'
            
            # Reset client to force re-initialization
            self.client = None
            
            self.logger.info(f"Saved API credentials to config file: {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config file: {str(e)}")
            return False
    
    def test_connection(self):
        """
        Test the API connection to ensure credentials are valid.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            client = self.get_client()
            # Try a simple API call to test connection
            account_info = client.get_user_principals()
            
            if account_info:
                self.logger.info("API connection test successful")
                return True
            else:
                self.logger.warning("API connection test failed: No data returned")
                return False
        except Exception as e:
            self.logger.error(f"API connection test failed: {str(e)}")
            return False
