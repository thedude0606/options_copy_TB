"""
Main entry point for the Schwab Options Dashboard application
"""
import os
import sys

# Add the current directory to the path to fix import issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app directly from the app directory
from app.app import app

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
