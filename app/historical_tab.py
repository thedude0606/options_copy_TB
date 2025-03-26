"""
Historical data tab functionality for options dashboard
"""
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
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
            id="historical-period",
            options=[
                {"label": "1 Day", "value": "day"},
                {"label": "1 Week", "value": "week"},
                {"label": "1 Month", "value": "month"},
                {"label": "3 Months", "value": "3month"},
                {"label": "1 Year", "value": "year"}
            ],
            value="month"
        ),
        
        # Frequency selection
        html.Label("Frequency:"),
        dcc.Dropdown(
            id="historical-frequency",
            options=[
                {"label": "1 Minute", "value": "1min"},
                {"label": "5 Minutes", "value": "5min"},
                {"label": "15 Minutes", "value": "15min"},
                {"label": "30 Minutes", "value": "30min"},
                {"label": "60 Minutes", "value": "60min"},
                {"label": "Daily", "value": "daily"}
            ],
            value="daily"
        ),
        
        # Load button
        html.Button(
            "Load Data",
            id="load-historical-data",
            style={"margin-top": "10px", "margin-bottom": "20px"}
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
        # Validate and correct common symbol mistakes
        symbol = symbol.upper().strip()
        
        # Common symbol corrections
        symbol_corrections = {
            'APPL': 'AAPL',  # Apple Inc.
            'GOGL': 'GOOGL',  # Alphabet Inc.
            'AMZM': 'AMZN',   # Amazon.com Inc.
            'MSFT': 'MSFT',   # Microsoft (already correct, included for completeness)
            'TSLA': 'TSLA',   # Tesla (already correct, included for completeness)
            'NVDA': 'NVDA',   # NVIDIA (already correct, included for completeness)
            'AMZN': 'AMZN',   # Amazon (already correct, included for completeness)
            'GOOGL': 'GOOGL', # Alphabet (already correct, included for completeness)
            'FB': 'META',     # Meta Platforms (formerly Facebook)
            'TWTR': 'TWTR',   # Twitter (already correct, included for completeness)
        }
        
        # Apply correction if needed
        if symbol in symbol_corrections:
            corrected_symbol = symbol_corrections[symbol]
            if symbol != corrected_symbol:
                print(f"Symbol corrected from {symbol} to {corrected_symbol}")
                symbol = corrected_symbol
        
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
            frequency=freq_value
        )
        
        print(f"Fetching historical data for {symbol} from {datetime.now() - timedelta(days=30)} to {datetime.now()}")
        print(f"Period: {time_period}, FrequencyType: {freq_type}, Frequency: {freq_value}")
        
        # Check if historical_data is None and handle it
        if historical_data is None:
            print("Historical data is None")
            return go.Figure()
            
        # Convert to DataFrame if it's not already
        if not isinstance(historical_data, pd.DataFrame):
            try:
                # Check if it's a list of dictionaries or similar structure
                if isinstance(historical_data, list) and len(historical_data) > 0:
                    print(f"historical_data type: {type(historical_data)}")
                    print(f"historical_data length: {len(historical_data)}")
                    
                    # Create DataFrame from list
                    df = pd.DataFrame(historical_data)
                    print(f"DataFrame created with shape: {df.shape}")
                    print(f"DataFrame columns: {df.columns.tolist()}")
                    print(f"DataFrame head: \n{df.head()}")
                    historical_data = df
                else:
                    print(f"historical_data type: {type(historical_data)}")
                    print(f"historical_data length: {len(historical_data) if hasattr(historical_data, '__len__') else 0}")
                    print("No historical data available")
                    return go.Figure()
            except Exception as e:
                print(f"Error converting historical data to DataFrame: {str(e)}")
                return go.Figure()
        
        if historical_data.empty:
            print("Historical data DataFrame is empty")
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
        
        print("Historical chart figure created successfully")
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
        [Input("load-historical-data", "n_clicks")],
        [State("symbol-input", "value"),
         State("historical-period", "value"),
         State("historical-frequency", "value")],
        prevent_initial_call=True
    )
    def update_chart_callback(n_clicks, symbol, period, frequency):
        if not n_clicks or not symbol:
            return go.Figure()
        
        # Map period to time_period format expected by update_historical_chart
        period_mapping = {
            'day': '1D',
            'week': '1W',
            'month': '1M',
            '3month': '3M',
            'year': '1Y'
        }
        
        time_period = period_mapping.get(period, '1M')
        
        # Override frequency if specified
        if frequency:
            freq_mapping = {
                '1min': ('minute', 1),
                '5min': ('minute', 5),
                '15min': ('minute', 15),
                '30min': ('minute', 30),
                '60min': ('minute', 60),
                'daily': ('daily', 1)
            }
            
            # Get period parameters from time_period
            period_mapping_params = {
                '1D': ('day', 1, 'minute', 5),
                '1W': ('day', 7, 'minute', 30),
                '1M': ('month', 1, 'daily', 1),
                '3M': ('month', 3, 'daily', 1),
                '1Y': ('year', 1, 'daily', 1)
            }
            
            period_type, period_value, _, _ = period_mapping_params.get(time_period, ('month', 1, 'daily', 1))
            
            if frequency in freq_mapping:
                freq_type, freq_value = freq_mapping[frequency]
                
                # Get historical data with custom frequency
                data_collector = DataCollector()
                historical_data = data_collector.get_historical_data(
                    symbol, 
                    period_type=period_type, 
                    period=period_value, 
                    frequency_type=freq_type, 
                    frequency=freq_value
                )
                
                # Check if historical_data is None and handle it
                if historical_data is None:
                    print("Historical data is None")
                    return go.Figure()
                    
                # Convert to DataFrame if it's not already
                if not isinstance(historical_data, pd.DataFrame):
                    try:
                        # Check if it's a list of dictionaries or similar structure
                        if isinstance(historical_data, list) and len(historical_data) > 0:
                            # Create DataFrame from list
                            historical_data = pd.DataFrame(historical_data)
                        else:
                            print("No historical data available")
                            return go.Figure()
                    except Exception as e:
                        print(f"Error converting historical data to DataFrame: {str(e)}")
                        return go.Figure()
                
                if historical_data.empty:
                    print("Historical data DataFrame is empty")
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
                    title=f"Historical Data for {symbol} ({period} - {frequency})",
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
        
        # If no custom frequency or if custom frequency failed, use default
        return update_historical_chart(symbol, time_period)
