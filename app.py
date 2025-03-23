"""
Main entry point for the Schwab Options Dashboard application
"""
import os
import sys

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.app import app

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
