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
                
                # Period selection (first, as it affects timeframe options)
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
                
                # Time frame selection
                html.Label("Time Frame:"),
                dbc.Select(
                    id="indicator-timeframe",
                    options=[
                        {"label": "Daily", "value": "1d"},
                        {"label": "Weekly", "value": "1wk"}
                    ],
                    value="1d",
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
                                    value=14,
                                    min=2,
                                    max=50,
                                    step=1,
                                    className="mb-2"
                                ),
                                html.Label("Overbought Level:"),
                                dbc.Input(
                                    id="rsi-overbought",
                                    type="number",
                                    value=70,
                                    min=50,
                                    max=95,
                                    step=1,
                                    className="mb-2"
                                ),
                                html.Label("Oversold Level:"),
                                dbc.Input(
                                    id="rsi-oversold",
                                    type="number",
                                    value=30,
                                    min=5,
                                    max=50,
                                    step=1,
                                    className="mb-3"
                                )
                            ]),
                            
                            # MACD parameters
                            html.Div([
                                html.H6("MACD Parameters", className="mb-2"),
                                html.Label("Fast Period:"),
                                dbc.Input(
                                    id="macd-fast",
                                    type="number",
                                    value=12,
                                    min=2,
                                    max=50,
                                    step=1,
                                    className="mb-2"
                                ),
                                html.Label("Slow Period:"),
                                dbc.Input(
                                    id="macd-slow",
                                    type="number",
                                    value=26,
                                    min=5,
                                    max=100,
                                    step=1,
                                    className="mb-2"
                                ),
                                html.Label("Signal Period:"),
                                dbc.Input(
                                    id="macd-signal",
                                    type="number",
                                    value=9,
                                    min=2,
                                    max=50,
                                    step=1,
                                    className="mb-3"
                                )
                            ]),
                            
                            # Bollinger Bands parameters
                            html.Div([
                                html.H6("Bollinger Bands Parameters", className="mb-2"),
                                html.Label("Period:"),
                                dbc.Input(
                                    id="bb-period",
                                    type="number",
                                    value=20,
                                    min=5,
                                    max=50,
                                    step=1,
                                    className="mb-2"
                                ),
                                html.Label("Standard Deviations:"),
                                dbc.Input(
                                    id="bb-std",
                                    type="number",
                                    value=2,
                                    min=0.5,
                                    max=4,
                                    step=0.1,
                                    className="mb-3"
                                )
                            ]),
                            
                            # Moving Averages parameters
                            html.Div([
                                html.H6("Moving Averages Parameters", className="mb-2"),
                                html.Label("SMA 1 Period:"),
                                dbc.Input(
                                    id="sma1-period",
                                    type="number",
                                    value=20,
                                    min=5,
                                    max=200,
                                    step=1,
                                    className="mb-2"
                                ),
                                html.Label("SMA 2 Period:"),
                                dbc.Input(
                                    id="sma2-period",
                                    type="number",
                                    value=50,
                                    min=10,
                                    max=200,
                                    step=1,
                                    className="mb-2"
                                ),
                                html.Label("EMA Period:"),
                                dbc.Input(
                                    id="ema-period",
                                    type="number",
                                    value=20,
                                    min=5,
                                    max=200,
                                    step=1,
                                    className="mb-3"
                                )
                            ]),
                            
                            # Reset button
                            dbc.Button(
                                "Reset to Defaults",
                                id="reset-params-button",
                                color="secondary",
                                size="sm",
                                className="mt-2"
                            )
                        ]),
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

# Callback to update timeframe options based on selected period
def update_timeframe_options(period):
    """
    Update timeframe options based on selected period to ensure compatibility
    
    Args:
        period (str): Selected period value
        
    Returns:
        tuple: Updated timeframe options and default value
    """
    # Define compatible timeframes for each period
    period_compatible_timeframes = {
        '1w': [
            {"label": "1 Minute", "value": "1m"},
            {"label": "5 Minutes", "value": "5m"},
            {"label": "15 Minutes", "value": "15m"},
            {"label": "1 Hour", "value": "1h"}
        ],
        '1m': [
            {"label": "Daily", "value": "1d"},
            {"label": "Weekly", "value": "1wk"}
        ],
        '3m': [
            {"label": "Daily", "value": "1d"},
            {"label": "Weekly", "value": "1wk"}
        ],
        '6m': [
            {"label": "Daily", "value": "1d"},
            {"label": "Weekly", "value": "1wk"}
        ],
        '1y': [
            {"label": "Daily", "value": "1d"},
            {"label": "Weekly", "value": "1wk"}
        ]
    }
    
    # Get compatible timeframes for selected period
    if period in period_compatible_timeframes:
        options = period_compatible_timeframes[period]
        # Set default value to first option
        default_value = options[0]["value"]
        return options, default_value
    
    # Default options if period not recognized
    default_options = [
        {"label": "Daily", "value": "1d"},
        {"label": "Weekly", "value": "1wk"}
    ]
    return default_options, "1d"

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
        period_type, period_value, default_freq_type, default_freq_value = period_mapping.get(period, ('month', 1, 'daily', 1))
        
        # Check compatibility between period type and frequency type
        compatible_freq_types = {
            'day': ['minute'],
            'month': ['daily', 'weekly'],
            'year': ['daily', 'weekly', 'monthly']
        }
        
        # Get timeframe parameters with better error handling
        if timeframe in timeframe_mapping:
            freq_type, freq_value = timeframe_mapping[timeframe]
            
            # Check if the selected frequency type is compatible with the period type
            if period_type in compatible_freq_types and freq_type not in compatible_freq_types[period_type]:
                print(f"Warning: Incompatible frequency_type '{freq_type}' for period_type '{period_type}'")
                print(f"Using default frequency type '{default_freq_type}' instead")
                freq_type, freq_value = default_freq_type, default_freq_value
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
        tech_indicators = TechnicalIndicators()
        
        # Create figure with subplots
        fig = make_subplots(rows=4, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.02, 
                           row_heights=[0.5, 0.2, 0.15, 0.15])
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=historical_data.index,
                open=historical_data['open'],
                high=historical_data['high'],
                low=historical_data['low'],
                close=historical_data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add volume as bar chart
        fig.add_trace(
            go.Bar(
                x=historical_data.index,
                y=historical_data['volume'],
                name="Volume",
                marker_color='rgba(0, 0, 255, 0.3)'
            ),
            row=2, col=1
        )
        
        # Add selected indicators
        if 'bollinger' in indicators:
            middle_band, upper_band, lower_band = tech_indicators.calculate_bollinger_bands(
                historical_data, period=bb_period, num_std=bb_std
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=middle_band,
                    name=f"BB Middle ({bb_period})",
                    line=dict(color='rgba(255, 165, 0, 0.7)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=upper_band,
                    name=f"BB Upper ({bb_period}, {bb_std}σ)",
                    line=dict(color='rgba(0, 128, 0, 0.7)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=lower_band,
                    name=f"BB Lower ({bb_period}, {bb_std}σ)",
                    line=dict(color='rgba(255, 0, 0, 0.7)')
                ),
                row=1, col=1
            )
        
        if 'ma' in indicators:
            # Calculate SMAs
            sma1 = tech_indicators.calculate_sma(historical_data, period=sma1_period)
            sma2 = tech_indicators.calculate_sma(historical_data, period=sma2_period)
            ema = tech_indicators.calculate_ema(historical_data, period=ema_period)
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=sma1,
                    name=f"SMA ({sma1_period})",
                    line=dict(color='rgba(255, 0, 255, 0.7)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=sma2,
                    name=f"SMA ({sma2_period})",
                    line=dict(color='rgba(0, 255, 255, 0.7)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=ema,
                    name=f"EMA ({ema_period})",
                    line=dict(color='rgba(128, 0, 128, 0.7)')
                ),
                row=1, col=1
            )
        
        if 'rsi' in indicators:
            # Calculate RSI
            rsi = tech_indicators.calculate_rsi(historical_data, period=rsi_period)
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=rsi,
                    name=f"RSI ({rsi_period})",
                    line=dict(color='rgba(75, 0, 130, 0.7)')
                ),
                row=3, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=[historical_data.index[0], historical_data.index[-1]],
                    y=[rsi_overbought, rsi_overbought],
                    name=f"Overbought ({rsi_overbought})",
                    line=dict(color='rgba(255, 0, 0, 0.5)', dash='dash')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[historical_data.index[0], historical_data.index[-1]],
                    y=[rsi_oversold, rsi_oversold],
                    name=f"Oversold ({rsi_oversold})",
                    line=dict(color='rgba(0, 128, 0, 0.5)', dash='dash')
                ),
                row=3, col=1
            )
            
            # Add midline
            fig.add_trace(
                go.Scatter(
                    x=[historical_data.index[0], historical_data.index[-1]],
                    y=[50, 50],
                    name="RSI Midline",
                    line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash')
                ),
                row=3, col=1
            )
        
        if 'macd' in indicators:
            # Calculate MACD
            macd_line, signal_line, histogram = tech_indicators.calculate_macd(
                historical_data, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=macd_line,
                    name=f"MACD ({macd_fast}, {macd_slow})",
                    line=dict(color='rgba(0, 0, 255, 0.7)')
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=signal_line,
                    name=f"Signal ({macd_signal})",
                    line=dict(color='rgba(255, 165, 0, 0.7)')
                ),
                row=4, col=1
            )
            
            # Add histogram as bar chart
            colors = ['rgba(0, 255, 0, 0.7)' if val >= 0 else 'rgba(255, 0, 0, 0.7)' for val in histogram]
            
            fig.add_trace(
                go.Bar(
                    x=historical_data.index,
                    y=histogram,
                    name="MACD Histogram",
                    marker_color=colors
                ),
                row=4, col=1
            )
        
        if 'mfi' in indicators:
            # Calculate MFI
            mfi = tech_indicators.calculate_mfi(historical_data)
            
            if not mfi.empty:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=mfi,
                        name="MFI",
                        line=dict(color='rgba(255, 69, 0, 0.7)')
                    ),
                    row=3, col=1
                )
        
        if 'imi' in indicators:
            # Calculate IMI
            imi = tech_indicators.calculate_imi(historical_data)
            
            if not imi.empty:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=imi,
                        name="IMI",
                        line=dict(color='rgba(0, 128, 128, 0.7)')
                    ),
                    row=3, col=1
                )
        
        if 'fvg' in indicators:
            # Calculate Fair Value Gap
            fvg = tech_indicators.calculate_fair_value_gap(historical_data)
            
            if not fvg.empty:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=fvg,
                        name="FVG",
                        line=dict(color='rgba(139, 0, 139, 0.7)')
                    ),
                    row=1, col=1
                )
        
        if 'liquidity' in indicators:
            # Calculate Liquidity Zones
            liquidity_zones = tech_indicators.calculate_liquidity_zones(historical_data)
            
            if not liquidity_zones.empty:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=liquidity_zones,
                        name="Liquidity Zones",
                        line=dict(color='rgba(0, 100, 0, 0.7)')
                    ),
                    row=1, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Technical Analysis ({period}, {timeframe})",
            xaxis_title="Date",
            yaxis_title="Price",
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        if 'rsi' in indicators or 'mfi' in indicators or 'imi' in indicators:
            fig.update_yaxes(title_text="Oscillators", row=3, col=1)
        
        if 'macd' in indicators:
            fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        # Update x-axis
        fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[16, 9.5], pattern="hour")  # hide non-trading hours
            ]
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating indicators chart: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return go.Figure()

# Register callbacks
def register_callbacks(app):
    """
    Register all callbacks for the indicators tab
    
    Args:
        app: Dash app instance
    """
    app.callback(
        Output("indicator-params-collapse", "is_open"),
        [Input("indicator-params-toggle", "n_clicks")],
        [State("indicator-params-collapse", "is_open")]
    )(toggle_indicator_params)
    
    app.callback(
        Output("rsi-period", "value"),
        Output("rsi-overbought", "value"),
        Output("rsi-oversold", "value"),
        Output("macd-fast", "value"),
        Output("macd-slow", "value"),
        Output("macd-signal", "value"),
        Output("bb-period", "value"),
        Output("bb-std", "value"),
        Output("sma1-period", "value"),
        Output("sma2-period", "value"),
        Output("ema-period", "value"),
        Input("reset-params-button", "n_clicks")
    )(reset_indicator_params)
    
    app.callback(
        Output("indicator-timeframe", "options"),
        Output("indicator-timeframe", "value"),
        Input("indicator-period", "value")
    )(update_timeframe_options)
    
    app.callback(
        Output("indicators-chart", "figure"),
        [
            Input("update-indicators-button", "n_clicks"),
            Input("indicator-analyze-button", "n_clicks")
        ],
        [
            State("indicator-symbol-input", "value"),
            State("indicator-timeframe", "value"),
            State("indicator-period", "value"),
            State("indicator-selection", "value"),
            State("rsi-period", "value"),
            State("rsi-overbought", "value"),
            State("rsi-oversold", "value"),
            State("macd-fast", "value"),
            State("macd-slow", "value"),
            State("macd-signal", "value"),
            State("bb-period", "value"),
            State("bb-std", "value"),
            State("sma1-period", "value"),
            State("sma2-period", "value"),
            State("ema-period", "value")
        ]
    )(update_indicators_chart)
