import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables
SCHWAB_APP_KEY = os.getenv('SCHWAB_APP_KEY')
SCHWAB_APP_SECRET = os.getenv('SCHWAB_APP_SECRET')
SCHWAB_CALLBACK_URL = os.getenv('SCHWAB_CALLBACK_URL')

# Create .env file if it doesn't exist
def create_env_file():
    """Create a .env file with the provided credentials if it doesn't exist"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(f"SCHWAB_APP_KEY={SCHWAB_APP_KEY}\n")
            f.write(f"SCHWAB_APP_SECRET={SCHWAB_APP_SECRET}\n")
            f.write(f"SCHWAB_CALLBACK_URL={SCHWAB_CALLBACK_URL}\n")
        print(".env file created successfully")
    else:
        print(".env file already exists")
