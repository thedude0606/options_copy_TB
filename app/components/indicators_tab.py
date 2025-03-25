"""
Technical indicators tab for options recommendation platform.
Implements the UI for displaying and analyzing technical indicators.
"""
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from app.indicators.technical_indicators import TechnicalIndicators
from app.data_collector import DataCollector

def create_indicators_tab():
    """
    Create the technical indicators tab layout
    
    Returns:
        html.Div: Technical indicators tab layout
    """
    return html.Div([
        dbc.Row([
            # Symbol input and controls
            dbc.Col([
                html.H4("Technical Indicators Analysis", className="mb-3"),
                html.Label("Symbol:"),
                dbc.InputGroup([
                    dbc.Input(id="indicator-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
                    dbc.Button("Analyze", id="indicator-analyze-button", color="primary")
                ], className="mb-3"),
                
                # Time frame selection
                html.Label("Time Frame:"),
                dbc.Select(
                    id="indicator-timeframe",
                    options=[
                        {"label": "1 Minute", "value": "1m"},
                        {"label": "5 Minutes", "value": "5m"},
                        {"label": "15 Minutes", "value": "15m"},
                        {"label": "1 Hour", "value": "1h"},
                        {"label": "Daily", "value": "1d"},
                        {"label": "Weekly", "value": "1wk"}
                    ],
                    value="1d",
                    className="mb-3"
                ),
                
                # Period selection
                html.Label("Period:"),
                dbc.Select(
                    id="indicator-period",
                    options=[
                        {"label": "1 Week", "value": "1w"},
                        {"label": "1 Month", "value": "1m"},
                        {"label": "3 Months", "value": "3m"},
                        {"label": "6 Months", "value": "6m"},
                        {"label": "1 Year", "value": "1y"}
                    ],
                    value="1m",
                    className="mb-3"
                ),
                
                # Indicator selection
                html.H5("Select Indicators", className="mb-2"),
                dbc.Checklist(
                    id="indicator-selection",
                    options=[
                        {"label": "RSI", "value": "rsi"},
                        {"label": "MACD", "value": "macd"},
                        {"label": "Bollinger Bands", "value": "bollinger"},
                        {"label": "Moving Averages", "value": "ma"},
                        {"label": "Money Flow Index (MFI)", "value": "mfi"},
                        {"label": "Intraday Momentum Index (IMI)", "value": "imi"},
                        {"label": "Fair Value Gap (FVG)", "value": "fvg"},
                        {"label": "Liquidity Zones", "value": "liquidity"}
                    ],
                    value=["rsi", "macd", "bollinger"],
                    className="mb-3"
                ),
                
                # Indicator parameters (collapsible)
                dbc.Button(
                    "Indicator Parameters",
                    id="indicator-params-toggle",
                    color="secondary",
                    outline=True,
                    size="sm",
                    className="mb-2"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody([
                            # RSI parameters
                            html.Div([
                                html.H6("RSI Parameters", className="mb-2"),
                                html.Label("RSI Period:"),
                                dbc.Input(
                                    id="rsi-period",
                                    type="number",
                                    min=2,
                                    max=30,
                                    step=1,
                                    value=14,
                                    className="mb-2"
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Overbought:"),
                                        dbc.Input(
                                            id="rsi-overbought",
                                            type="number",
                                            min=50,
                                            max=90,
                                            step=1,
                                            value=70
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Oversold:"),
                                        dbc.Input(
                                            id="rsi-oversold",
                                            type="number",
                                            min=10,
                                            max=50,
                                            step=1,
                                            value=30
                                        )
                                    ], width=6)
                                ], className="mb-3")
                            ]),
                            
                            # MACD parameters
                            html.Div([
                                html.H6("MACD Parameters", className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Fast Period:"),
                                        dbc.Input(
                                            id="macd-fast-period",
                                            type="number",
                                            min=5,
                                            max=20,
                                            step=1,
                                            value=12
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Slow Period:"),
                                        dbc.Input(
                                            id="macd-slow-period",
                                            type="number",
                                            min=10,
                                            max=40,
                                            step=1,
                                            value=26
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Signal Period:"),
                                        dbc.Input(
                                            id="macd-signal-period",
                                            type="number",
                                            min=2,
                                            max=15,
                                            step=1,
                                            value=9
                                        )
                                    ], width=4)
                                ], className="mb-3")
                            ]),
                            
                            # Bollinger Bands parameters
                            html.Div([
                                html.H6("Bollinger Bands Parameters", className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Period:"),
                                        dbc.Input(
                                            id="bb-period",
                                            type="number",
                                            min=5,
                                            max=50,
                                            step=1,
                                            value=20
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Standard Deviations:"),
                                        dbc.Input(
                                            id="bb-std",
                                            type="number",
                                            min=1,
                                            max=4,
                                            step=0.5,
                                            value=2
                                        )
                                    ], width=6)
                                ], className="mb-3")
                            ]),
                            
                            # Moving Averages parameters
                            html.Div([
                                html.H6("Moving Averages Parameters", className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("SMA 1:"),
                                        dbc.Input(
                                            id="sma1-period",
                                            type="number",
                                            min=5,
                                            max=100,
                                            step=1,
                                            value=20
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("SMA 2:"),
                                        dbc.Input(
                                            id="sma2-period",
                                            type="number",
                                            min=5,
                                            max=200,
                                            step=1,
                                            value=50
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("EMA:"),
                                        dbc.Input(
                                            id="ema-period",
                                            type="number",
                                            min=5,
                                            max=100,
                                            step=1,
                                            value=20
                                        )
                                    ], width=4)
                                ], className="mb-3")
                            ]),
                            
                            # Reset button
                            dbc.Button(
                                "Reset to Defaults",
                                id="reset-indicator-params",
                                color="secondary",
                                size="sm",
                                className="mt-2"
                            )
                        ])
                    ),
                    id="indicator-params-collapse",
                    is_open=False
                ),
                
                # Update button
                dbc.Button(
                    "Update Chart",
                    id="update-indicators-button",
                    color="primary",
                    className="mt-3"
                )
            ], width=3),
            
            # Chart area
            dbc.Col([
                html.Div(id="indicators-chart-container", children=[
                    dcc.Loading(
                        id="indicators-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id="indicators-chart", style={"height": "800px"})
                        ]
                    )
                ])
            ], width=9)
        ])
    ])

# Callback to toggle indicator parameters collapse
def toggle_indicator_params(n_clicks, is_open):
    """
    Toggle the indicator parameters collapse
    
    Args:
        n_clicks (int): Number of clicks
        is_open (bool): Current state of collapse
        
    Returns:
        bool: New state of collapse
    """
    if n_clicks:
        return not is_open
    return is_open

# Callback to reset indicator parameters to defaults
def reset_indicator_params(n_clicks):
    """
    Reset indicator parameters to default values
    
    Args:
        n_clicks (int): Number of clicks
        
    Returns:
        tuple: Default values for all parameters
    """
    if n_clicks:
        return 14, 70, 30, 12, 26, 9, 20, 2, 20, 50, 20
    
    # If not triggered by button click, return no update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
           dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
           dash.no_update, dash.no_update, dash.no_update

# Callback to update indicators chart
def update_indicators_chart(update_clicks, analyze_clicks, 
                           symbol, timeframe, period, indicators,
                           rsi_period, rsi_overbought, rsi_oversold,
                           macd_fast, macd_slow, macd_signal,
                           bb_period, bb_std, sma1_period, sma2_period, ema_period):
    """
    Update the technical indicators chart
    
    Args:
        update_clicks (int): Number of update button clicks
        analyze_clicks (int): Number of analyze button clicks
        symbol (str): Stock symbol
        timeframe (str): Time frame for data
        period (str): Period for data
        indicators (list): Selected indicators
        rsi_period (int): RSI period
        rsi_overbought (int): RSI overbought level
        rsi_oversold (int): RSI oversold level
        macd_fast (int): MACD fast period
        macd_slow (int): MACD slow period
        macd_signal (int): MACD signal period
        bb_period (int): Bollinger Bands period
        bb_std (float): Bollinger Bands standard deviations
        sma1_period (int): SMA 1 period
        sma2_period (int): SMA 2 period
        ema_period (int): EMA period
        
    Returns:
        go.Figure: Technical indicators chart
    """
    # Check if callback was triggered
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure()
    
    # Get trigger ID
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Only update if triggered by buttons
    if trigger_id not in ['update-indicators-button', 'indicator-analyze-button']:
        return dash.no_update
    
    # Validate symbol
    if not symbol:
        return go.Figure()
    
    try:
        # Map period to API parameters with improved mapping
        period_mapping = {
            '1w': ('day', 7, 'minute', 30),
            '1m': ('month', 1, 'daily', 1),
            '3m': ('month', 3, 'daily', 1),
            '6m': ('month', 6, 'daily', 1),
            '1y': ('year', 1, 'daily', 1)
        }
        
        # Map timeframe to API parameters with correct frequency values
        timeframe_mapping = {
            '1m': ('minute', 1),
            '5m': ('minute', 5),
            '15m': ('minute', 15),
            '1h': ('minute', 60),
            '1d': ('daily', 1),
            '1wk': ('weekly', 1)
        }
        
        # Get period parameters with better error handling
        period_type, period_value, freq_type, freq_value = period_mapping.get(period, ('month', 1, 'daily', 1))
        
        # Get timeframe parameters with better error handling
        if timeframe in timeframe_mapping:
            freq_type, freq_value = timeframe_mapping[timeframe]
        else:
            print(f"Warning: Unknown timeframe '{timeframe}', defaulting to daily")
            freq_type, freq_value = 'daily', 1
            
        print(f"Using period_type={period_type}, period_value={period_value}, freq_type={freq_type}, freq_value={freq_value}")
        
        # Get historical data
        data_collector = DataCollector()
        historical_data = data_collector.get_historical_data(
            symbol, 
            period_type=period_type, 
            period=period_value, 
            frequency_type=freq_type, 
            frequency=freq_value
        )
        
        if historical_data.empty:
            return go.Figure()
        
        # Create technical indicators
        ti = TechnicalIndicators(historical_data)
        
        # Create subplots
        fig = make_subplots(
            rows=4, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=("Price", "Volume", "", "")
        )
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=historical_data.index if hasattr(historical_data, 'index') else historical_data['datetime'],
                open=historical_data['open'],
                high=historical_data['high'],
                low=historical_data['low'],
                close=historical_data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add volume chart
        fig.add_trace(
            go.Bar(
                x=historical_data.index if hasattr(historical_data, 'index') else historical_data['datetime'],
                y=historical_data['volume'],
                name="Volume",
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=2, col=1
        )
        
        # Add selected indicators
        subplot_row = 3  # Start additional indicators from row 3
        
        # Add Bollinger Bands
        if 'bollinger' in indicators:
            bb = ti.bollinger_bands(period=bb_period, std_dev=bb_std)
            
            fig.add_trace(
                go.Scatter(
                    x=bb.index if hasattr(bb, 'index') else historical_data['datetime'],
                    y=bb['middle_band'],
                    name=f"BB Middle ({bb_period})",
                    line=dict(color='rgba(0, 0, 255, 0.7)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=bb.index if hasattr(bb, 'index') else historical_data['datetime'],
                    y=bb['upper_band'],
                    name=f"BB Upper ({bb_period}, {bb_std}σ)",
                    line=dict(color='rgba(0, 0, 255, 0.3)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=bb.index if hasattr(bb, 'index') else historical_data['datetime'],
                    y=bb['lower_band'],
                    name=f"BB Lower ({bb_period}, {bb_std}σ)",
                    line=dict(color='rgba(0, 0, 255, 0.3)'),
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # Add Moving Averages
        if 'ma' in indicators:
            sma1 = ti.sma(period=sma1_period)
            sma2 = ti.sma(period=sma2_period)
            ema = ti.ema(period=ema_period)
            
            fig.add_trace(
                go.Scatter(
                    x=sma1.index if hasattr(sma1, 'index') else historical_data['datetime'],
                    y=sma1,
                    name=f"SMA ({sma1_period})",
                    line=dict(color='rgba(255, 0, 0, 0.7)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sma2.index if hasattr(sma2, 'index') else historical_data['datetime'],
                    y=sma2,
                    name=f"SMA ({sma2_period})",
                    line=dict(color='rgba(0, 255, 0, 0.7)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=ema.index if hasattr(ema, 'index') else historical_data['datetime'],
                    y=ema,
                    name=f"EMA ({ema_period})",
                    line=dict(color='rgba(255, 165, 0, 0.7)')
                ),
                row=1, col=1
            )
        
        # Add RSI
        if 'rsi' in indicators:
            rsi = ti.rsi(period=rsi_period)
            
            fig.add_trace(
                go.Scatter(
                    x=rsi.index if hasattr(rsi, 'index') else historical_data['datetime'],
                    y=rsi,
                    name=f"RSI ({rsi_period})",
                    line=dict(color='purple')
                ),
                row=subplot_row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=rsi.index if hasattr(rsi, 'index') else historical_data['datetime'],
                    y=[rsi_overbought] * len(rsi),
                    name=f"Overbought ({rsi_overbought})",
                    line=dict(color='red', dash='dash')
                ),
                row=subplot_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rsi.index if hasattr(rsi, 'index') else historical_data['datetime'],
                    y=[rsi_oversold] * len(rsi),
                    name=f"Oversold ({rsi_oversold})",
                    line=dict(color='green', dash='dash')
                ),
                row=subplot_row, col=1
            )
            
            # Update y-axis range
            fig.update_yaxes(range=[0, 100], row=subplot_row, col=1, title_text="RSI")
            
            subplot_row += 1
        
        # Add MACD
        if 'macd' in indicators:
            macd_data = ti.macd(fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
            
            fig.add_trace(
                go.Scatter(
                    x=macd_data.index if hasattr(macd_data, 'index') else historical_data['datetime'],
                    y=macd_data['macd'],
                    name=f"MACD ({macd_fast},{macd_slow},{macd_signal})",
                    line=dict(color='blue')
                ),
                row=subplot_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=macd_data.index if hasattr(macd_data, 'index') else historical_data['datetime'],
                    y=macd_data['signal'],
                    name=f"Signal ({macd_signal})",
                    line=dict(color='red')
                ),
                row=subplot_row, col=1
            )
            
            # Add histogram
            fig.add_trace(
                go.Bar(
                    x=macd_data.index if hasattr(macd_data, 'index') else historical_data['datetime'],
                    y=macd_data['histogram'],
                    name="Histogram",
                    marker_color='rgba(0, 255, 0, 0.5)'
                ),
                row=subplot_row, col=1
            )
            
            fig.update_yaxes(title_text="MACD", row=subplot_row, col=1)
            
            subplot_row += 1
        
        # Add MFI
        if 'mfi' in indicators and subplot_row <= 4:
            mfi = ti.money_flow_index(period=14)
            
            fig.add_trace(
                go.Scatter(
                    x=mfi.index if hasattr(mfi, 'index') else historical_data['datetime'],
                    y=mfi,
                    name="MFI (14)",
                    line=dict(color='orange')
                ),
                row=subplot_row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=mfi.index if hasattr(mfi, 'index') else historical_data['datetime'],
                    y=[80] * len(mfi),
                    name="Overbought (80)",
                    line=dict(color='red', dash='dash')
                ),
                row=subplot_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=mfi.index if hasattr(mfi, 'index') else historical_data['datetime'],
                    y=[20] * len(mfi),
                    name="Oversold (20)",
                    line=dict(color='green', dash='dash')
                ),
                row=subplot_row, col=1
            )
            
            fig.update_yaxes(range=[0, 100], title_text="MFI", row=subplot_row, col=1)
            
            subplot_row += 1
        
        # Update layout
        fig.update_layout(
            title=f"Technical Analysis for {symbol}",
            xaxis_title="Date",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=800,
            margin=dict(l=50, r=50, t=80, b=50)
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
        print(f"Error updating indicators chart: {str(e)}")
        return go.Figure()

def register_indicators_callbacks(app):
    """
    Register callback functions for the indicators tab
    
    Args:
        app: The Dash app instance
    """
    # Callback to toggle indicator parameters collapse
    @app.callback(
        Output("indicator-params-collapse", "is_open"),
        [Input("indicator-params-toggle", "n_clicks")],
        [State("indicator-params-collapse", "is_open")]
    )
    def toggle_params_callback(n_clicks, is_open):
        return toggle_indicator_params(n_clicks, is_open)
    
    # Callback to reset indicator parameters to defaults
    @app.callback(
        [Output("rsi-period", "value"),
         Output("rsi-overbought", "value"),
         Output("rsi-oversold", "value"),
         Output("macd-fast-period", "value"),
         Output("macd-slow-period", "value"),
         Output("macd-signal-period", "value"),
         Output("bb-period", "value"),
         Output("bb-std", "value"),
         Output("sma1-period", "value"),
         Output("sma2-period", "value"),
         Output("ema-period", "value")],
        [Input("reset-indicator-params", "n_clicks")]
    )
    def reset_params_callback(n_clicks):
        return reset_indicator_params(n_clicks)
    
    # Callback to update indicators chart
    @app.callback(
        Output("indicators-chart", "figure"),
        [Input("update-indicators-button", "n_clicks"),
         Input("indicator-analyze-button", "n_clicks")],
        [State("indicator-symbol-input", "value"),
         State("indicator-timeframe", "value"),
         State("indicator-period", "value"),
         State("indicator-selection", "value"),
         State("rsi-period", "value"),
         State("rsi-overbought", "value"),
         State("rsi-oversold", "value"),
         State("macd-fast-period", "value"),
         State("macd-slow-period", "value"),
         State("macd-signal-period", "value"),
         State("bb-period", "value"),
         State("bb-std", "value"),
         State("sma1-period", "value"),
         State("sma2-period", "value"),
         State("ema-period", "value")]
    )
    def update_chart_callback(update_clicks, analyze_clicks, 
                             symbol, timeframe, period, indicators,
                             rsi_period, rsi_overbought, rsi_oversold,
                             macd_fast, macd_slow, macd_signal,
                             bb_period, bb_std, sma1_period, sma2_period, ema_period):
        return update_indicators_chart(update_clicks, analyze_clicks, 
                                      symbol, timeframe, period, indicators,
                                      rsi_period, rsi_overbought, rsi_oversold,
                                      macd_fast, macd_slow, macd_signal,
                                      bb_period, bb_std, sma1_period, sma2_period, ema_period)
