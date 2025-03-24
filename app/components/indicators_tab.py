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
                                        html.Label("Standard Deviation:"),
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
                                dbc.Checklist(
                                    id="ma-selection",
                                    options=[
                                        {"label": "SMA 20", "value": "sma20"},
                                        {"label": "SMA 50", "value": "sma50"},
                                        {"label": "SMA 200", "value": "sma200"},
                                        {"label": "EMA 9", "value": "ema9"},
                                        {"label": "EMA 21", "value": "ema21"}
                                    ],
                                    value=["sma20", "sma50"],
                                    className="mb-3"
                                )
                            ])
                        ])
                    ),
                    id="indicator-params-collapse",
                    is_open=False
                ),
                
                # Action buttons
                dbc.Button("Update Chart", id="update-indicators-button", color="success", className="mr-2 mt-3"),
                dbc.Button("Reset Parameters", id="reset-indicators-button", color="secondary", outline=True, className="mt-3")
            ], md=3, className="sidebar"),
            
            # Chart display area
            dbc.Col([
                # Loading spinner for chart
                dbc.Spinner(
                    dcc.Graph(id="indicators-chart", style={"height": "700px"}),
                    color="primary",
                    type="border",
                    fullscreen=False
                ),
                
                # Status messages
                html.Div(id="indicators-status", className="status-message mt-3")
            ], md=9)
        ])
    ])

# Callback for toggling indicator parameters
@callback(
    Output("indicator-params-collapse", "is_open"),
    Input("indicator-params-toggle", "n_clicks"),
    State("indicator-params-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_indicator_params(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback for resetting indicator parameters
@callback(
    [Output("rsi-period", "value"),
     Output("rsi-overbought", "value"),
     Output("rsi-oversold", "value"),
     Output("macd-fast-period", "value"),
     Output("macd-slow-period", "value"),
     Output("macd-signal-period", "value"),
     Output("bb-period", "value"),
     Output("bb-std", "value"),
     Output("ma-selection", "value")],
    Input("reset-indicators-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_indicator_params(n_clicks):
    return 14, 70, 30, 12, 26, 9, 20, 2, ["sma20", "sma50"]

# Callback for updating indicators chart
@callback(
    [Output("indicators-chart", "figure"),
     Output("indicators-status", "children")],
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
     State("ma-selection", "value")],
    prevent_initial_call=True
)
def update_indicators_chart(update_clicks, analyze_clicks, 
                           symbol, timeframe, period, 
                           selected_indicators, rsi_period, 
                           rsi_overbought, rsi_oversold,
                           macd_fast, macd_slow, macd_signal,
                           bb_period, bb_std, ma_selection):
    # Check if any button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure(), ""
    
    # Validate inputs
    if not symbol:
        return go.Figure(), html.Div("Please enter a symbol", className="text-danger")
    
    try:
        # Map timeframe to Schwab API parameters
        timeframe_mapping = {
            "1m": {"frequency_type": "minute", "frequency": 1},
            "5m": {"frequency_type": "minute", "frequency": 5},
            "15m": {"frequency_type": "minute", "frequency": 15},
            "1h": {"frequency_type": "minute", "frequency": 60},
            "1d": {"frequency_type": "daily", "frequency": 1},
            "1wk": {"frequency_type": "weekly", "frequency": 1}
        }
        
        period_mapping = {
            "1w": {"period_type": "day", "period": 5},
            "1m": {"period_type": "month", "period": 1},
            "3m": {"period_type": "month", "period": 3},
            "6m": {"period_type": "month", "period": 6},
            "1y": {"period_type": "year", "period": 1}
        }
        
        # Initialize data collector and technical indicators
        data_collector = DataCollector(interactive_auth=False)
        tech_indicators = TechnicalIndicators()
        
        # Get historical data
        historical_data = data_collector.get_historical_data(
            symbol=symbol.upper(),
            period_type=period_mapping[period]["period_type"],
            period=period_mapping[period]["period"],
            frequency_type=timeframe_mapping[timeframe]["frequency_type"],
            frequency=timeframe_mapping[timeframe]["frequency"]
        )
        
        if historical_data.empty:
            return go.Figure(), html.Div(f"No historical data available for {symbol.upper()}", className="text-warning")
        
        # Determine number of subplots based on selected indicators
        n_rows = 1  # Main price chart
        if "rsi" in selected_indicators:
            n_rows += 1
        if "macd" in selected_indicators:
            n_rows += 1
        if "mfi" in selected_indicators:
            n_rows += 1
        if "imi" in selected_indicators:
            n_rows += 1
        
        # Create subplot grid
        row_heights = [0.5]  # Main price chart gets 50% of height
        remaining_height = 0.5
        for _ in range(n_rows - 1):
            row_heights.append(remaining_height / (n_rows - 1))
        
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=row_heights,
                           subplot_titles=["Price Chart"] + [""] * (n_rows - 1))
        
        # Add candlestick chart
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
        
        # Add Bollinger Bands
        if "bollinger" in selected_indicators:
            middle_band, upper_band, lower_band = tech_indicators.calculate_bollinger_bands(
                historical_data, window=bb_period, num_std=bb_std
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=upper_band,
                    name="Upper BB",
                    line=dict(color="rgba(173, 216, 230, 0.7)"),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=middle_band,
                    name="Middle BB",
                    line=dict(color="rgba(173, 216, 230, 1.0)"),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=lower_band,
                    name="Lower BB",
                    line=dict(color="rgba(173, 216, 230, 0.7)"),
                    fill="tonexty",
                    fillcolor="rgba(173, 216, 230, 0.2)",
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add Moving Averages
        if "ma" in selected_indicators:
            ma_dict = tech_indicators.calculate_moving_averages(historical_data)
            
            ma_colors = {
                "SMA_20": "rgba(255, 165, 0, 0.8)",  # Orange
                "SMA_50": "rgba(75, 192, 192, 0.8)",  # Teal
                "SMA_200": "rgba(153, 102, 255, 0.8)",  # Purple
                "EMA_9": "rgba(255, 99, 132, 0.8)",  # Red
                "EMA_21": "rgba(54, 162, 235, 0.8)"   # Blue
            }
            
            ma_mapping = {
                "sma20": "SMA_20",
                "sma50": "SMA_50",
                "sma200": "SMA_200",
                "ema9": "EMA_9",
                "ema21": "EMA_21"
            }
            
            for ma_key in ma_selection:
                ma_name = ma_mapping[ma_key]
                if ma_name in ma_dict and not ma_dict[ma_name].empty:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=ma_dict[ma_name],
                            name=ma_name.replace("_", " "),
                            line=dict(color=ma_colors.get(ma_name, "rgba(0, 0, 0, 0.8)")),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        
        # Add Fair Value Gap
        if "fvg" in selected_indicators:
            fvg_data = tech_indicators.calculate_fair_value_gap(historical_data)
            
            if not fvg_data.empty:
                for _, fvg in fvg_data.iterrows():
                    color = "rgba(0, 255, 0, 0.2)" if fvg['type'] == 'bullish' else "rgba(255, 0, 0, 0.2)"
                    
                    fig.add_shape(
                        type="rect",
                        x0=fvg['datetime'],
                        y0=fvg['low'],
                        x1=fvg['datetime'] + pd.Timedelta(days=5),  # Extend 5 days
                        y1=fvg['high'],
                        fillcolor=color,
                        opacity=0.5,
                        layer="below",
                        line=dict(width=0),
                        row=1, col=1
                    )
        
        # Add Liquidity Zones
        if "liquidity" in selected_indicators:
            liquidity_zones = tech_indicators.calculate_liquidity_zones(historical_data)
            
            if not liquidity_zones.empty:
                for _, zone in liquidity_zones.iterrows():
                    color = "rgba(0, 128, 0, 0.3)" if zone['type'] == 'support' else "rgba(139, 0, 0, 0.3)"
                    
                    fig.add_shape(
                        type="line",
                        x0=min(historical_data.index),
                        y0=zone['price'],
                        x1=max(historical_data.index),
                        y1=zone['price'],
                        line=dict(color=color, width=2, dash="dash"),
                        row=1, col=1
                    )
                    
                    fig.add_annotation(
                        x=max(historical_data.index),
                        y=zone['price'],
                        text=f"{zone['type'].capitalize()} ({zone['strength']:.1f})",
                        showarrow=False,
                        xanchor="right",
                        font=dict(color="black", size=10),
                        bgcolor=color,
                        bordercolor="black",
                        borderwidth=1,
                        row=1, col=1
                    )
        
        # Add RSI
        current_row = 2
        if "rsi" in selected_indicators:
            rsi = tech_indicators.calculate_rsi(historical_data, window=rsi_period)
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=rsi,
                    name="RSI",
                    line=dict(color="purple"),
                    showlegend=True
                ),
                row=current_row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=min(historical_data.index),
                y0=rsi_overbought,
                x1=max(historical_data.index),
                y1=rsi_overbought,
                line=dict(color="red", width=1, dash="dash"),
                row=current_row, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=min(historical_data.index),
                y0=rsi_oversold,
                x1=max(historical_data.index),
                y1=rsi_oversold,
                line=dict(color="green", width=1, dash="dash"),
                row=current_row, col=1
            )
            
            # Add annotations
            fig.add_annotation(
                x=max(historical_data.index),
                y=rsi_overbought,
                text="Overbought",
                showarrow=False,
                xanchor="right",
                font=dict(color="red", size=10),
                row=current_row, col=1
            )
            
            fig.add_annotation(
                x=max(historical_data.index),
                y=rsi_oversold,
                text="Oversold",
                showarrow=False,
                xanchor="right",
                font=dict(color="green", size=10),
                row=current_row, col=1
            )
            
            # Update y-axis range
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
            
            current_row += 1
        
        # Add MACD
        if "macd" in selected_indicators:
            macd_line, signal_line, histogram = tech_indicators.calculate_macd(
                historical_data, 
                fast_period=macd_fast, 
                slow_period=macd_slow, 
                signal_period=macd_signal
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=macd_line,
                    name="MACD",
                    line=dict(color="blue"),
                    showlegend=True
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=signal_line,
                    name="Signal",
                    line=dict(color="red"),
                    showlegend=True
                ),
                row=current_row, col=1
            )
            
            # Add histogram
            colors = ['green' if val >= 0 else 'red' for val in histogram]
            
            fig.add_trace(
                go.Bar(
                    x=historical_data.index,
                    y=histogram,
                    name="Histogram",
                    marker_color=colors,
                    showlegend=True
                ),
                row=current_row, col=1
            )
            
            # Update y-axis title
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
            
            current_row += 1
        
        # Add MFI
        if "mfi" in selected_indicators and current_row <= n_rows:
            mfi = tech_indicators.calculate_mfi(historical_data)
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=mfi,
                    name="MFI",
                    line=dict(color="orange"),
                    showlegend=True
                ),
                row=current_row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=min(historical_data.index),
                y0=80,
                x1=max(historical_data.index),
                y1=80,
                line=dict(color="red", width=1, dash="dash"),
                row=current_row, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=min(historical_data.index),
                y0=20,
                x1=max(historical_data.index),
                y1=20,
                line=dict(color="green", width=1, dash="dash"),
                row=current_row, col=1
            )
            
            # Update y-axis range
            fig.update_yaxes(title_text="MFI", range=[0, 100], row=current_row, col=1)
            
            current_row += 1
        
        # Add IMI
        if "imi" in selected_indicators and current_row <= n_rows:
            imi = tech_indicators.calculate_imi(historical_data)
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=imi,
                    name="IMI",
                    line=dict(color="brown"),
                    showlegend=True
                ),
                row=current_row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=min(historical_data.index),
                y0=70,
                x1=max(historical_data.index),
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
                row=current_row, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=min(historical_data.index),
                y0=30,
                x1=max(historical_data.index),
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
                row=current_row, col=1
            )
            
            # Update y-axis range
            fig.update_yaxes(title_text="IMI", range=[0, 100], row=current_row, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol.upper()} Technical Analysis",
            xaxis_rangeslider_visible=False,
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
        
        # Update x-axis
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[16, 9.5], pattern="hour")  # Hide non-trading hours
            ]
        )
        
        return fig, html.Div(f"Technical analysis for {symbol.upper()} completed successfully", className="text-success")
    
    except Exception as e:
        return go.Figure(), html.Div(f"Error generating technical analysis: {str(e)}", className="text-danger")
