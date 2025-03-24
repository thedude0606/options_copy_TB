"""
Historical data tab functionality for options dashboard
"""
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from app.data_collector import DataCollector

def create_historical_tab():
    """
    Create the historical data tab layout
    
    Returns:
        html.Div: Historical data tab layout
    """
    return html.Div([
        # Time period selection
        html.Label("Time Period:"),
        dcc.Dropdown(
            id="time-period",
            options=[
                {"label": "1 Day", "value": "1D"},
                {"label": "1 Week", "value": "1W"},
                {"label": "1 Month", "value": "1M"},
                {"label": "3 Months", "value": "3M"},
                {"label": "1 Year", "value": "1Y"}
            ],
            value="1M"
        ),
        
        # Candle chart
        dcc.Graph(id="historical-chart")
    ])

def update_historical_chart(symbol, time_period):
    """
    Update the historical data chart
    
    Args:
        symbol (str): Stock symbol
        time_period (str): Time period for data
        
    Returns:
        go.Figure: Historical data chart
    """
    if not symbol:
        return go.Figure()
    
    try:
        # Map time period to API parameters
        period_mapping = {
            '1D': ('day', 1, 'minute', 5),
            '1W': ('day', 7, 'minute', 30),
            '1M': ('month', 1, 'daily', 1),
            '3M': ('month', 3, 'daily', 1),
            '1Y': ('year', 1, 'daily', 1)
        }
        
        # Get period parameters
        period_type, period_value, freq_type, freq_value = period_mapping.get(time_period, ('month', 1, 'daily', 1))
        
        # Get historical data
        data_collector = DataCollector()
        historical_data = data_collector.get_historical_data(
            symbol, 
            period_type=period_type, 
            period=period_value, 
            frequency_type=freq_type, 
            frequency=frequency_value
        )
        
        if historical_data.empty:
            return go.Figure()
        
        # Create candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=historical_data.index if hasattr(historical_data, 'index') else historical_data['datetime'],
                open=historical_data['open'],
                high=historical_data['high'],
                low=historical_data['low'],
                close=historical_data['close'],
                name="Price"
            )
        ])
        
        # Add volume as bar chart on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=historical_data.index if hasattr(historical_data, 'index') else historical_data['datetime'],
                y=historical_data['volume'],
                name="Volume",
                yaxis="y2",
                marker_color='rgba(0, 0, 255, 0.5)'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Historical Data for {symbol} ({time_period})",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Hide weekends and after hours
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[16, 9.5], pattern="hour")  # Hide after hours
            ]
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating historical chart: {str(e)}")
        return go.Figure()

def register_historical_callbacks(app):
    """
    Register callback functions for the historical data tab
    
    Args:
        app: The Dash app instance
    """
    @app.callback(
        Output("historical-chart", "figure"),
        [Input("submit-button", "n_clicks")],
        [State("symbol-input", "value"),
         State("time-period", "value")]
    )
    def update_chart_callback(n_clicks, symbol, time_period):
        if not n_clicks or not symbol:
            return go.Figure()
        
        return update_historical_chart(symbol, time_period)
