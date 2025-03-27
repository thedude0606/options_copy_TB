"""
Debug module for the trading timeline selector functionality.
This script adds visible debugging to help track when the timeline selector is used.
"""
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
import sys

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.indicators.technical_indicators import TechnicalIndicators
from app.data_collector import DataCollector

def add_debug_callback_to_app(app):
    """
    Add a debug callback to track timeframe selector changes
    
    Args:
        app: The Dash app instance
    """
    @app.callback(
        Output("debug-output", "children"),
        Input("indicator-timeframe", "value"),
        prevent_initial_call=True
    )
    def debug_timeframe_change(timeframe):
        """
        Debug callback to track timeframe selector changes
        
        Args:
            timeframe (str): Selected timeframe value
            
        Returns:
            html.Div: Debug information
        """
        return html.Div([
            html.H5("Timeframe Selector Debug Info", className="mt-3"),
            html.P(f"Timeframe changed to: {timeframe}"),
            html.P(f"Timestamp: {pd.Timestamp.now()}"),
            html.Hr()
        ], style={"border": "1px solid red", "padding": "10px", "margin": "10px 0"})

def patch_update_indicators_chart():
    """
    Patch the update_indicators_chart function to add visible debugging
    """
    # Import the original function
    from app.components.indicators_tab import update_indicators_chart as original_function
    
    # Define the patched function
    def patched_update_indicators_chart(*args, **kwargs):
        """
        Patched version of update_indicators_chart with visible debugging
        """
        # Extract timeframe from args (it's the 4th argument)
        timeframe = args[3] if len(args) > 3 else None
        
        # Create debug figure with timeframe info
        debug_fig = go.Figure()
        debug_fig.add_annotation(
            text=f"DEBUG: Using timeframe={timeframe}",
            xref="paper", yref="paper",
            x=0.5, y=0.9,
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="yellow",
            bordercolor="red",
            borderwidth=2
        )
        
        # Call the original function
        result_fig = original_function(*args, **kwargs)
        
        # Add debug annotation to the result figure
        result_fig.add_annotation(
            text=f"DEBUG: Using timeframe={timeframe}",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="yellow",
            bordercolor="red",
            borderwidth=2
        )
        
        return result_fig
    
    # Replace the original function with the patched one
    import app.components.indicators_tab
    app.components.indicators_tab.update_indicators_chart = patched_update_indicators_chart

def add_debug_div_to_layout():
    """
    Add a debug div to the indicators tab layout
    
    Returns:
        html.Div: Debug div to add to the layout
    """
    return html.Div([
        html.H4("Timeline Selector Debug Information", className="mt-4"),
        html.Div(id="debug-output"),
        dbc.Alert(
            [
                html.H5("Debugging Instructions"),
                html.P("1. Select a different timeframe from the dropdown above"),
                html.P("2. Click the 'Update' button to apply the change"),
                html.P("3. Check this area for debug information"),
                html.P("4. The chart should display a yellow debug annotation showing the selected timeframe")
            ],
            color="info",
            className="mt-2"
        )
    ])

def patch_data_collector():
    """
    Patch the DataCollector class to add more visible debugging
    """
    # Import the original class
    from app.data_collector import DataCollector as OriginalDataCollector
    
    # Define the patched method
    def patched_get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', 
                                   frequency=1, need_extended_hours_data=True):
        """
        Patched version of get_historical_data with enhanced debugging
        """
        # Create debug log file
        with open('debug_timeframe.log', 'a') as f:
            f.write(f"\n=== HISTORICAL DATA REQUEST ===\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Parameters: periodType={period_type}, period={period}, frequencyType={frequency_type}, frequency={frequency}\n")
        
        # Call the original method
        result = OriginalDataCollector.get_historical_data(self, symbol, period_type, period, 
                                                         frequency_type, frequency, need_extended_hours_data)
        
        # Log the result shape
        with open('debug_timeframe.log', 'a') as f:
            f.write(f"Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}\n")
            f.write(f"Result head: {result.head(2) if hasattr(result, 'head') else 'N/A'}\n")
            f.write(f"=== END REQUEST ===\n")
        
        return result
    
    # Replace the original method with the patched one
    import app.data_collector
    app.data_collector.DataCollector.get_historical_data = patched_get_historical_data

def install_debugging():
    """
    Install all debugging patches
    """
    # Patch the DataCollector
    patch_data_collector()
    
    # Patch the update_indicators_chart function
    patch_update_indicators_chart()
    
    # Add a note to the run_dashboard.py file to add the debug div and callback
    with open('debug_installation.txt', 'w') as f:
        f.write("""
DEBUGGING INSTALLATION INSTRUCTIONS

To complete the debugging installation, follow these steps:

1. In run_dashboard.py, add the following import at the top:
   from tests.debug_timeline_selector import add_debug_callback_to_app, add_debug_div_to_layout

2. After creating the app (app = dash.Dash(...)), add:
   add_debug_callback_to_app(app)

3. In the indicators_tab.py file, find the create_indicators_tab function and add the debug div:
   - Find the return html.Div([...]) statement
   - Before the closing ]), add:
     add_debug_div_to_layout()

4. Run the dashboard and test the timeframe selector

5. Check the debug_timeframe.log file for detailed information
""")

if __name__ == "__main__":
    install_debugging()
    print("Debugging installation instructions written to debug_installation.txt")
