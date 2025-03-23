"""
Test script to verify Schwab API authentication
"""
import sys
import os
import webbrowser

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.auth import get_client
from dotenv import load_dotenv

def test_authentication():
    """Test authentication with Schwab API"""
    print("Testing Schwab API authentication...")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get client with interactive mode enabled
        client = get_client(interactive=True)
        
        # Test a simple API call to verify authentication
        accounts = client.account_linked()
        
        if accounts:
            print("Authentication successful!")
            print(f"Found {len(accounts)} linked accounts")
            return True
        else:
            print("Authentication successful, but no accounts found")
            return True
            
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_authentication()
