"""
Real-time data tab component for the options dashboard
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import pandas as pd
import json
from datetime import datetime
from app.data_collector import DataCollector

# Define the layout for the real-time data tab
def get_real_time_tab_layout():
    """
    Create the layout for the real-time data tab
    
    Returns:
        html.Div: The layout component
    """
    return html.Div([
        # Controls section
        html.Div([
            html.H3("Real-Time Data Controls"),
            
            # Symbol selection
            html.Div([
                html.Label("Symbol:"),
                dcc.Input(
                    id="rt-symbol-input",
                    type="text",
                    value="",
                    placeholder="Enter symbol (e.g., AAPL)",
                    style={"width": "150px", "margin-right": "10px"}
                ),
                html.Button(
                    "Add Symbol",
                    id="rt-add-symbol-button",
                    n_clicks=0,
                    style={"margin-right": "10px"}
                ),
                
                # Display active symbols
                html.Div([
                    html.Label("Active Symbols:"),
                    html.Div(id="rt-active-symbols", style={"margin-top": "5px"})
                ], style={"margin-top": "10px"})
            ], style={"margin-bottom": "20px"}),
            
            # Connection controls
            html.Div([
                html.Button(
                    "Start Stream",
                    id="rt-start-stream-button",
                    n_clicks=0,
                    style={"margin-right": "10px"}
                ),
                html.Button(
                    "Stop Stream",
                    id="rt-stop-stream-button",
                    n_clicks=0,
                    style={"margin-right": "10px"}
                ),
                html.Div(id="rt-connection-status", style={"margin-top": "5px"})
            ], style={"margin-bottom": "20px"}),
            
            # Data type selection
            html.Div([
                html.Label("Data Type:"),
                dcc.RadioItems(
                    id="rt-data-type",
                    options=[
                        {"label": "Quotes", "value": "quotes"},
                        {"label": "Options", "value": "options"}
                    ],
                    value="quotes",
                    inline=True
                )
            ], style={"margin-bottom": "20px"})
        ], style={"padding": "15px", "background-color": "#f8f9fa", "border-radius": "5px"}),
        
        # Data display section
        html.Div([
            html.H3("Real-Time Data"),
            
            # Tabs for different data views
            dcc.Tabs([
                # Price chart tab
                dcc.Tab(label="Price Chart", children=[
                    dcc.Graph(id="rt-price-chart")
                ]),
                
                # Data table tab
                dcc.Tab(label="Data Table", children=[
                    html.Div(id="rt-data-table")
                ]),
                
                # Time & Sales tab
                dcc.Tab(label="Time & Sales", children=[
                    html.Div(id="rt-time-sales")
                ])
            ])
        ], style={"margin-top": "20px"}),
        
        # Hidden divs for storing data
        dcc.Store(id="rt-stream-data"),
        dcc.Store(id="rt-symbols-store"),
        dcc.Store(id="rt-connection-store"),
        
        # Interval component for updating charts
        dcc.Interval(
            id="rt-update-interval",
            interval=1000,  # 1 second
            n_intervals=0,
            disabled=True
        )
    ])

# Define callback functions for the real-time data tab
def register_real_time_callbacks(app):
    """
    Register callback functions for the real-time data tab
    
    Args:
        app: The Dash app instance
    """
    
    # Callback to add symbols
    @app.callback(
        [Output("rt-symbols-store", "data"),
         Output("rt-active-symbols", "children")],
        [Input("rt-add-symbol-button", "n_clicks")],
        [State("rt-symbol-input", "value"),
         State("rt-symbols-store", "data")]
    )
    def add_symbol(n_clicks, symbol, symbols_data):
        if n_clicks == 0:
            return [], html.Div("No symbols added")
        
        if not symbol:
            return symbols_data or [], html.Div("No symbols added") if not symbols_data else html.Div(", ".join(symbols_data))
        
        # Initialize symbols list if needed
        if not symbols_data:
            symbols_data = []
        
        # Add symbol if not already in list
        if symbol.upper() not in [s.upper() for s in symbols_data]:
            symbols_data.append(symbol.upper())
        
        # Create display of active symbols
        symbols_display = html.Div([
            html.Span(", ".join(symbols_data)),
            html.Button(
                "Clear All",
                id="rt-clear-symbols-button",
                n_clicks=0,
                style={"margin-left": "10px", "font-size": "12px"}
            )
        ])
        
        return symbols_data, symbols_display
    
    # Callback to clear symbols
    @app.callback(
        [Output("rt-symbols-store", "data", allow_duplicate=True),
         Output("rt-active-symbols", "children", allow_duplicate=True)],
        [Input("rt-clear-symbols-button", "n_clicks")],
        prevent_initial_call=True
    )
    def clear_symbols(n_clicks):
        if n_clicks > 0:
            return [], html.Div("No symbols added")
        return dash.no_update, dash.no_update
    
    # Callback to start/stop stream
    @app.callback(
        [Output("rt-connection-status", "children"),
         Output("rt-connection-store", "data"),
         Output("rt-update-interval", "disabled")],
        [Input("rt-start-stream-button", "n_clicks"),
         Input("rt-stop-stream-button", "n_clicks")],
        [State("rt-connection-store", "data")]
    )
    def manage_stream(start_clicks, stop_clicks, connection_data):
        # Initialize connection data if needed
        if not connection_data:
            connection_data = {"active": False, "timestamp": None}
        
        # Determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            # Initial load
            status = html.Div([
                html.Span("Disconnected", style={"color": "red"}),
                html.Span(" - Stream not started", style={"font-style": "italic", "margin-left": "5px"})
            ])
            return status, connection_data, True
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "rt-start-stream-button" and start_clicks > 0:
            # Start stream
            connection_data = {
                "active": True,
                "timestamp": datetime.now().isoformat()
            }
            status = html.Div([
                html.Span("Connected", style={"color": "green"}),
                html.Span(f" - Started at {connection_data['timestamp']}", style={"font-style": "italic", "margin-left": "5px"})
            ])
            return status, connection_data, False
        
        elif button_id == "rt-stop-stream-button" and stop_clicks > 0:
            # Stop stream
            connection_data = {
                "active": False,
                "timestamp": datetime.now().isoformat()
            }
            status = html.Div([
                html.Span("Disconnected", style={"color": "red"}),
                html.Span(f" - Stopped at {connection_data['timestamp']}", style={"font-style": "italic", "margin-left": "5px"})
            ])
            return status, connection_data, True
        
        # Default return
        if connection_data["active"]:
            status = html.Div([
                html.Span("Connected", style={"color": "green"}),
                html.Span(f" - Started at {connection_data['timestamp']}", style={"font-style": "italic", "margin-left": "5px"})
            ])
            return status, connection_data, False
        else:
            status = html.Div([
                html.Span("Disconnected", style={"color": "red"}),
                html.Span(" - Stream not started", style={"font-style": "italic", "margin-left": "5px"})
            ])
            return status, connection_data, True
    
    # Callback to update stream data
    @app.callback(
        Output("rt-stream-data", "data"),
        [Input("rt-update-interval", "n_intervals")],
        [State("rt-symbols-store", "data"),
         State("rt-stream-data", "data"),
         State("rt-connection-store", "data"),
         State("rt-data-type", "value")]
    )
    def update_stream_data(n_intervals, symbols, stream_data, connection_data, data_type):
        # Check if stream is active
        if not connection_data or not connection_data.get("active", False):
            return stream_data or {}
        
        # Check if we have symbols to fetch
        if not symbols:
            return stream_data or {}
        
        # Initialize stream data if needed
        if not stream_data:
            stream_data = {}
        
        try:
            # Initialize data collector
            data_collector = DataCollector()
            
            # Current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Fetch data for each symbol
            for symbol in symbols:
                # Initialize symbol data if needed
                if symbol not in stream_data:
                    stream_data[symbol] = []
                
                # Fetch real-time data
                if data_type == "quotes":
                    # Fetch quote data
                    quote_data = data_collector.get_real_time_data(symbol)
                    
                    if quote_data:
                        # Extract relevant fields
                        data_point = {
                            "timestamp": timestamp,
                            "price": quote_data.get("lastPrice", None),
                            "change": quote_data.get("netChange", None),
                            "percent_change": quote_data.get("netPercentChangeInDouble", None),
                            "bid": quote_data.get("bidPrice", None),
                            "ask": quote_data.get("askPrice", None),
                            "volume": quote_data.get("totalVolume", None),
                            "size": quote_data.get("lastSize", None)
                        }
                        
                        # Add to stream data (limit to last 100 points)
                        stream_data[symbol].append(data_point)
                        if len(stream_data[symbol]) > 100:
                            stream_data[symbol] = stream_data[symbol][-100:]
                else:  # options
                    # Fetch options data
                    options_data = data_collector.get_option_chain_with_underlying_price(symbol)
                    
                    if options_data and "underlyingPrice" in options_data:
                        # Extract relevant fields
                        data_point = {
                            "timestamp": timestamp,
                            "price": options_data.get("underlyingPrice", None),
                            "change": None,  # Not available in options data
                            "percent_change": None,  # Not available in options data
                            "bid": None,  # Would need specific option contract
                            "ask": None,  # Would need specific option contract
                            "volume": None,  # Would need specific option contract
                            "open_interest": None,  # Would need specific option contract
                            "size": None  # Not available in options data
                        }
                        
                        # Add to stream data (limit to last 100 points)
                        stream_data[symbol].append(data_point)
                        if len(stream_data[symbol]) > 100:
                            stream_data[symbol] = stream_data[symbol][-100:]
            
            return stream_data
        
        except Exception as e:
            print(f"Error updating stream data: {str(e)}")
            return stream_data or {}
    
    # Callback to update price chart
    @app.callback(
        Output("rt-price-chart", "figure"),
        [Input("rt-update-interval", "n_intervals"),
         Input("rt-stream-data", "data")],
        [State("rt-symbols-store", "data")]
    )
    def update_price_chart(n_intervals, stream_data, symbols):
        # Create empty figure if no data
        if not stream_data or not symbols:
            print(f"DEBUG - update_price_chart: No data or symbols available. stream_data={bool(stream_data)}, symbols={symbols}")
            fig = go.Figure()
            fig.update_layout(
                title="Real-Time Price Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                showlegend=True
            )
            return fig
        
        # Parse stream data
        try:
            data_dict = stream_data
            print(f"DEBUG - update_price_chart: Processing data for symbols: {symbols}")
            print(f"DEBUG - update_price_chart: Available symbols in data: {list(data_dict.keys())}")
            
            # Create figure
            fig = go.Figure()
            
            # Add trace for each symbol
            for symbol in symbols:
                if symbol in data_dict:
                    symbol_data = data_dict[symbol]
                    print(f"DEBUG - update_price_chart: Found {len(symbol_data)} data points for {symbol}")
                    
                    if not symbol_data:
                        print(f"DEBUG - update_price_chart: No data points for {symbol}")
                        continue
                    
                    # Extract time and price data
                    times = []
                    prices = []
                    
                    for item in symbol_data:
                        if isinstance(item, dict) and "timestamp" in item and "price" in item:
                            if item["price"] is not None:
                                times.append(item["timestamp"])
                                prices.append(float(item["price"]))
                    
                    # Only add trace if we have valid data
                    if prices and times:
                        print(f"DEBUG - update_price_chart: Adding trace for {symbol} with {len(prices)} price points")
                        # Add line trace
                        fig.add_trace(go.Scatter(
                            x=times,
                            y=prices,
                            mode="lines+markers",
                            name=symbol
                        ))
                    else:
                        print(f"DEBUG - update_price_chart: No valid data for {symbol}, skipping trace")
                else:
                    print(f"DEBUG - update_price_chart: Symbol {symbol} not found in data_dict")
            
            # Update layout
            fig.update_layout(
                title="Real-Time Price Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                showlegend=True
            )
            
            return fig
        
        except Exception as e:
            # Return empty figure on error
            print(f"DEBUG - update_price_chart: Error processing data: {str(e)}")
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                xaxis_title="Time",
                yaxis_title="Price",
                showlegend=True
            )
            return fig
    
    # Callback to update data table
    @app.callback(
        Output("rt-data-table", "children"),
        [Input("rt-update-interval", "n_intervals"),
         Input("rt-stream-data", "data")],
        [State("rt-symbols-store", "data"),
         State("rt-data-type", "value")]
    )
    def update_data_table(n_intervals, stream_data, symbols, data_type):
        if not stream_data or not symbols:
            print(f"DEBUG - update_data_table: No data or symbols available. stream_data={bool(stream_data)}, symbols={symbols}")
            return html.Div("No data available")
        
        try:
            data_dict = stream_data
            print(f"DEBUG - update_data_table: Processing data for symbols: {symbols}")
            print(f"DEBUG - update_data_table: Available symbols in data: {list(data_dict.keys())}")
            
            # Create table header based on data type
            if data_type == "quotes":
                header = html.Tr([
                    html.Th("Symbol"),
                    html.Th("Last Price"),
                    html.Th("Change"),
                    html.Th("% Change"),
                    html.Th("Bid"),
                    html.Th("Ask"),
                    html.Th("Volume"),
                    html.Th("Last Update")
                ])
            else:  # options
                header = html.Tr([
                    html.Th("Symbol"),
                    html.Th("Last Price"),
                    html.Th("Change"),
                    html.Th("% Change"),
                    html.Th("Bid"),
                    html.Th("Ask"),
                    html.Th("Volume"),
                    html.Th("Open Interest"),
                    html.Th("Last Update")
                ])
            
            # Create table rows
            rows = []
            for symbol in symbols:
                if symbol in data_dict and data_dict[symbol]:
                    # Get latest data point
                    latest = data_dict[symbol][-1]
                    
                    if not isinstance(latest, dict):
                        print(f"DEBUG - update_data_table: Latest data for {symbol} is not a dictionary: {type(latest)}")
                        continue
                    
                    # Create row based on data type
                    if data_type == "quotes":
                        row = html.Tr([
                            html.Td(symbol),
                            html.Td(f"${latest.get('price')}" if latest.get('price') is not None else "N/A"),
                            html.Td(f"{latest.get('change')}" if latest.get('change') is not None else "N/A"),
                            html.Td(f"{latest.get('percent_change')}%" if latest.get('percent_change') is not None else "N/A"),
                            html.Td(f"${latest.get('bid')}" if latest.get('bid') is not None else "N/A"),
                            html.Td(f"${latest.get('ask')}" if latest.get('ask') is not None else "N/A"),
                            html.Td(f"{latest.get('volume')}" if latest.get('volume') is not None else "N/A"),
                            html.Td(latest.get('timestamp', 'N/A'))
                        ])
                    else:  # options
                        row = html.Tr([
                            html.Td(symbol),
                            html.Td(f"${latest.get('price')}" if latest.get('price') is not None else "N/A"),
                            html.Td(f"{latest.get('change')}" if latest.get('change') is not None else "N/A"),
                            html.Td(f"{latest.get('percent_change')}%" if latest.get('percent_change') is not None else "N/A"),
                            html.Td(f"${latest.get('bid')}" if latest.get('bid') is not None else "N/A"),
                            html.Td(f"${latest.get('ask')}" if latest.get('ask') is not None else "N/A"),
                            html.Td(f"{latest.get('volume')}" if latest.get('volume') is not None else "N/A"),
                            html.Td(f"{latest.get('open_interest')}" if latest.get('open_interest') is not None else "N/A"),
                            html.Td(latest.get('timestamp', 'N/A'))
                        ])
                    
                    rows.append(row)
                else:
                    print(f"DEBUG - update_data_table: No data available for symbol {symbol}")
            
            # Create table
            table = html.Table(
                [header] + rows,
                style={"width": "100%", "border-collapse": "collapse"}
            )
            
            print(f"DEBUG - update_data_table: Created table with {len(rows)} rows")
            return table
        
        except Exception as e:
            print(f"DEBUG - update_data_table: Error processing data: {str(e)}")
            return html.Div(f"Error: {str(e)}")
    
    # Callback to update time & sales
    @app.callback(
        Output("rt-time-sales", "children"),
        [Input("rt-update-interval", "n_intervals"),
         Input("rt-stream-data", "data")],
        [State("rt-symbols-store", "data")]
    )
    def update_time_sales(n_intervals, stream_data, symbols):
        if not stream_data or not symbols:
            return html.Div("No data available")
        
        try:
            data_dict = stream_data
            
            # Create list items for time & sales
            items = []
            for symbol in symbols:
                if symbol in data_dict:
                    symbol_data = data_dict[symbol]
                    
                    # Add last 10 data points in reverse order (newest first)
                    for data_point in reversed(symbol_data[-10:]):
                        # Format the time & sales entry
                        time_str = data_point.get("timestamp", "N/A")
                        price = data_point.get("price", "N/A")
                        size = data_point.get("size", "N/A")
                        
                        # Determine color based on price change
                        color = "inherit"
                        if "change" in data_point and data_point["change"] is not None:
                            if float(data_point["change"]) > 0:
                                color = "green"
                            elif float(data_point["change"]) < 0:
                                color = "red"
                        
                        # Create list item
                        item = html.Li([
                            html.Span(f"{time_str} - ", style={"color": "gray"}),
                            html.Span(f"{symbol}: ", style={"font-weight": "bold"}),
                            html.Span(f"${price} ", style={"color": color}),
                            html.Span(f"({size})")
                        ])
                        
                        items.append(item)
            
            # Create time & sales list
            if items:
                return html.Div([
                    html.H4("Recent Trades"),
                    html.Ul(items, style={"list-style-type": "none", "padding-left": "0"})
                ])
            else:
                return html.Div("No trade data available")
        
        except Exception as e:
            return html.Div(f"Error: {str(e)}")
