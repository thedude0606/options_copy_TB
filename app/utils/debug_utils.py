"""
Debug utilities for the options dashboard.
Provides debugging tools and utilities for the dashboard components.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import json
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger('debug_utils')
logger.setLevel(logging.DEBUG)

def add_debug_div_to_layout():
    """
    Create a debug div to display debugging information.
    
    Returns:
        html.Div: Debug div component
    """
    return html.Div([
        html.H4("Debug Information", className="mt-4"),
        html.Div(id="debug-output", className="border p-3 bg-light"),
        html.Div([
            html.Button("Clear Debug", id="clear-debug-button", className="btn btn-secondary mt-2")
        ])
    ], id="debug-container", className="mt-4")

def add_debug_callback_to_app(app):
    """
    Add debugging callbacks to the app.
    
    Args:
        app: The Dash app
    """
    @app.callback(
        dash.Output("debug-output", "children", allow_duplicate=True),
        [dash.Input("clear-debug-button", "n_clicks")],
        prevent_initial_call=True
    )
    def clear_debug(n_clicks):
        """
        Clear the debug output.
        
        Args:
            n_clicks: Number of button clicks
            
        Returns:
            html.Div: Empty div
        """
        return html.Div("Debug cleared")
    
    # Add a callback to capture and display errors
    app.clientside_callback(
        """
        function(errors) {
            if (errors) {
                return JSON.stringify(errors, null, 2);
            }
            return "No errors";
        }
        """,
        dash.Output("debug-output", "children", allow_duplicate=True),
        [dash.Input("_dash-app-content", "errors")],
        prevent_initial_call=True
    )
    
    logger.debug("Debug callbacks registered")

def debug_print(message, data=None):
    """
    Print debug information to the console.
    
    Args:
        message (str): Debug message
        data: Optional data to print
    """
    logger.debug(message)
    if data is not None:
        if isinstance(data, pd.DataFrame):
            logger.debug(f"DataFrame shape: {data.shape}")
            logger.debug(f"DataFrame columns: {data.columns.tolist()}")
            logger.debug(f"DataFrame head:\n{data.head()}")
        else:
            try:
                logger.debug(f"Data: {json.dumps(data, indent=2)}")
            except:
                logger.debug(f"Data: {str(data)}")

def format_dataframe_for_display(df, max_rows=10):
    """
    Format a DataFrame for display in the dashboard.
    
    Args:
        df (pandas.DataFrame): DataFrame to format
        max_rows (int): Maximum number of rows to display
        
    Returns:
        html.Div: Formatted DataFrame display
    """
    if df is None or df.empty:
        return html.Div("No data available")
    
    # Limit rows
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        footer = html.Div(f"Showing {max_rows} of {len(df)} rows", className="text-muted small")
    else:
        df_display = df
        footer = None
    
    # Create table header
    header = html.Thead(html.Tr([html.Th(col) for col in df_display.columns]))
    
    # Create table body
    rows = []
    for _, row in df_display.iterrows():
        rows.append(html.Tr([html.Td(row[col]) for col in df_display.columns]))
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table([header, body], bordered=True, hover=True, responsive=True, striped=True)
    
    # Return table with optional footer
    if footer:
        return html.Div([table, footer])
    else:
        return html.Div(table)

def create_debug_modal():
    """
    Create a debug modal for displaying detailed information.
    
    Returns:
        dbc.Modal: Debug modal component
    """
    return dbc.Modal(
        [
            dbc.ModalHeader("Debug Information"),
            dbc.ModalBody(id="debug-modal-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-debug-modal", className="ml-auto")
            ),
        ],
        id="debug-modal",
        size="lg",
    )

def register_debug_modal_callbacks(app):
    """
    Register callbacks for the debug modal.
    
    Args:
        app: The Dash app
    """
    @app.callback(
        dash.Output("debug-modal", "is_open"),
        [dash.Input("open-debug-modal", "n_clicks"), 
         dash.Input("close-debug-modal", "n_clicks")],
        [dash.State("debug-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_modal(open_clicks, close_clicks, is_open):
        """
        Toggle the debug modal.
        
        Args:
            open_clicks: Number of open button clicks
            close_clicks: Number of close button clicks
            is_open: Current modal state
            
        Returns:
            bool: New modal state
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "open-debug-modal":
                return True
            elif button_id == "close-debug-modal":
                return False
            return is_open
