"""
Simple entry point for the Schwab Options Dashboard application
"""
import os
import sys
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# App layout - simplified version for testing
app.layout = html.Div([
    html.H1("Schwab Options Dashboard"),
    
    # Symbol input and submit button
    html.Div([
        html.Label("Enter Symbol:"),
        dcc.Input(id="symbol-input", type="text", value="AAPL", placeholder="Enter stock symbol"),
        html.Button("Submit", id="submit-button", n_clicks=0),
    ], style={"margin": "20px"}),
    
    html.Div(id="output-container")
])

# Simple callback for testing
@app.callback(
    Output("output-container", "children"),
    [Input("submit-button", "n_clicks")],
    [State("symbol-input", "value")]
)
def update_output(n_clicks, symbol):
    if n_clicks == 0:
        return "Enter a symbol and click Submit"
    
    return f"You entered: {symbol}"

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
