"""
Validation visualizations for options recommendations.
Provides visual confirmation of technical patterns and trading signals.
"""
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_validation_chart(validation_data):
    """
    Create a validation chart for a recommendation
    
    Args:
        validation_data (dict): Validation data from recommendation engine
        
    Returns:
        dict: Plotly figure object
    """
    # Extract data
    symbol = validation_data.get('symbol', '')
    timeframe = validation_data.get('timeframe', '30m')
    option_type = validation_data.get('option_type', 'CALL')
    price_data = pd.DataFrame(validation_data.get('price_data', []))
    key_levels = validation_data.get('key_levels', {})
    indicators = validation_data.get('indicators', {})
    recommendation = validation_data.get('recommendation', {})
    
    if price_data.empty:
        return create_empty_chart("No price data available for validation")
    
    # Convert datetime to proper format if needed
    if 'datetime' in price_data.columns and not pd.api.types.is_datetime64_any_dtype(price_data['datetime']):
        price_data['datetime'] = pd.to_datetime(price_data['datetime'])
    
    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(
            f"{symbol} Price Action ({timeframe})",
            "Volume",
            "Technical Indicators"
        )
    )
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_data['datetime'],
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name="Price",
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add key price levels
    current_price = key_levels.get('current_price', 0)
    strike_price = key_levels.get('strike_price', 0)
    target_price = key_levels.get('target_price', 0)
    support_level = key_levels.get('support_level', 0)
    resistance_level = key_levels.get('resistance_level', 0)
    
    # Add horizontal lines for key levels
    if current_price > 0:
        fig.add_hline(
            y=current_price, 
            line_dash="solid", 
            line_color="blue", 
            annotation_text="Current Price",
            annotation_position="right",
            row=1, col=1
        )
    
    if strike_price > 0:
        fig.add_hline(
            y=strike_price, 
            line_dash="dash", 
            line_color="purple", 
            annotation_text="Strike Price",
            annotation_position="right",
            row=1, col=1
        )
    
    if target_price > 0:
        fig.add_hline(
            y=target_price, 
            line_dash="dot", 
            line_color="green" if option_type == "CALL" else "red", 
            annotation_text="Target Price",
            annotation_position="right",
            row=1, col=1
        )
    
    if support_level > 0:
        fig.add_hline(
            y=support_level, 
            line_dash="dot", 
            line_color="rgba(0,128,0,0.5)", 
            annotation_text="Support",
            annotation_position="left",
            row=1, col=1
        )
    
    if resistance_level > 0:
        fig.add_hline(
            y=resistance_level, 
            line_dash="dot", 
            line_color="rgba(255,0,0,0.5)", 
            annotation_text="Resistance",
            annotation_position="left",
            row=1, col=1
        )
    
    # Add Bollinger Bands if available
    bb_data = pd.DataFrame(indicators.get('bollinger_bands', []))
    if not bb_data.empty and 'datetime' in bb_data.columns:
        if not pd.api.types.is_datetime64_any_dtype(bb_data['datetime']):
            bb_data['datetime'] = pd.to_datetime(bb_data['datetime'])
            
        fig.add_trace(
            go.Scatter(
                x=bb_data['datetime'],
                y=bb_data['bb_upper'],
                mode='lines',
                line=dict(width=1, color='rgba(173, 216, 230, 0.7)'),
                name='Upper BB'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bb_data['datetime'],
                y=bb_data['bb_middle'],
                mode='lines',
                line=dict(width=1, color='rgba(173, 216, 230, 0.7)'),
                name='Middle BB'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bb_data['datetime'],
                y=bb_data['bb_lower'],
                mode='lines',
                line=dict(width=1, color='rgba(173, 216, 230, 0.7)'),
                name='Lower BB',
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.1)'
            ),
            row=1, col=1
        )
    
    # Add volume chart
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for _, row in price_data.iterrows()]
    fig.add_trace(
        go.Bar(
            x=price_data['datetime'],
            y=price_data['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Add RSI if available
    rsi_data = pd.DataFrame(indicators.get('rsi', []))
    if not rsi_data.empty and 'datetime' in rsi_data.columns:
        if not pd.api.types.is_datetime64_any_dtype(rsi_data['datetime']):
            rsi_data['datetime'] = pd.to_datetime(rsi_data['datetime'])
            
        fig.add_trace(
            go.Scatter(
                x=rsi_data['datetime'],
                y=rsi_data['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1)
    
    # Add MACD if available
    macd_data = pd.DataFrame(indicators.get('macd', []))
    if not macd_data.empty and 'datetime' in macd_data.columns and len(rsi_data) == 0:
        # Only add MACD if RSI is not shown (to avoid cluttering)
        if not pd.api.types.is_datetime64_any_dtype(macd_data['datetime']):
            macd_data['datetime'] = pd.to_datetime(macd_data['datetime'])
            
        fig.add_trace(
            go.Scatter(
                x=macd_data['datetime'],
                y=macd_data['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=macd_data['datetime'],
                y=macd_data['macd_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=3, col=1
        )
        
        # Add MACD histogram
        colors = ['#26a69a' if val >= 0 else '#ef5350' for val in macd_data['macd_hist']]
        fig.add_trace(
            go.Bar(
                x=macd_data['datetime'],
                y=macd_data['macd_hist'],
                name="MACD Hist",
                marker_color=colors
            ),
            row=3, col=1
        )
    
    # Add annotations for key signals
    signals = recommendation.get('technical_signals', {})
    market_signal = signals.get('market_signal', 'neutral')
    signal_strength = signals.get('signal_strength', 0)
    
    # Format signal text
    signal_text = f"Signal: {market_signal.upper()} (Strength: {signal_strength:.2f})"
    signal_color = "green" if market_signal == "bullish" else "red" if market_signal == "bearish" else "gray"
    
    # Add signal annotation
    fig.add_annotation(
        x=price_data['datetime'].iloc[0],
        y=price_data['high'].max(),
        text=signal_text,
        showarrow=False,
        font=dict(color=signal_color, size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=signal_color,
        borderwidth=1,
        borderpad=4,
        xanchor="left",
        yanchor="top",
        row=1, col=1
    )
    
    # Add recommendation details
    confidence = recommendation.get('confidence_score', 0)
    risk_reward = recommendation.get('risk_reward_ratio', 0)
    potential_return = recommendation.get('potential_return', 0)
    
    details_text = (
        f"Recommendation: {option_type} @ ${key_levels.get('strike_price', 0):.2f}<br>"
        f"Confidence: {confidence:.2f}<br>"
        f"Risk/Reward: {risk_reward:.2f}x<br>"
        f"Potential Return: {potential_return*100:.1f}%"
    )
    
    fig.add_annotation(
        x=price_data['datetime'].iloc[-1],
        y=price_data['high'].max(),
        text=details_text,
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        xanchor="right",
        yanchor="top",
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} {option_type} Option Recommendation Validation ({timeframe})",
        xaxis_title="Time",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        template="plotly_white",
        showlegend=False,  # Hide legend to save space
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update y-axis for RSI
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    # Update y-axis for volume
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_timeframe_comparison_chart(validation_data_by_timeframe):
    """
    Create a comparison chart showing signals across different timeframes
    
    Args:
        validation_data_by_timeframe (dict): Validation data for multiple timeframes
        
    Returns:
        dict: Plotly figure object
    """
    timeframes = validation_data_by_timeframe.keys()
    if not timeframes:
        return create_empty_chart("No timeframe data available")
    
    # Create subplot figure with one row per timeframe
    fig = make_subplots(
        rows=len(timeframes), 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{tf.upper()} Timeframe" for tf in timeframes]
    )
    
    # Add data for each timeframe
    for i, (timeframe, data) in enumerate(validation_data_by_timeframe.items(), 1):
        price_data = pd.DataFrame(data.get('price_data', []))
        if price_data.empty:
            continue
            
        # Convert datetime to proper format if needed
        if 'datetime' in price_data.columns and not pd.api.types.is_datetime64_any_dtype(price_data['datetime']):
            price_data['datetime'] = pd.to_datetime(price_data['datetime'])
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data['datetime'],
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name=f"Price ({timeframe})",
                increasing_line_color='#26a69a', 
                decreasing_line_color='#ef5350'
            ),
            row=i, col=1
        )
        
        # Add key price levels
        key_levels = data.get('key_levels', {})
        current_price = key_levels.get('current_price', 0)
        
        if current_price > 0:
            fig.add_hline(
                y=current_price, 
                line_dash="solid", 
                line_color="blue", 
                annotation_text="Current",
                annotation_position="right",
                row=i, col=1
            )
        
        # Add signal annotation
        recommendation = data.get('recommendation', {})
        signals = recommendation.get('technical_signals', {})
        market_signal = signals.get('market_signal', 'neutral')
        signal_strength = signals.get('signal_strength', 0)
        
        signal_color = "green" if market_signal == "bullish" else "red" if market_signal == "bearish" else "gray"
        
        fig.add_annotation(
            x=price_data['datetime'].iloc[0],
            y=price_data['high'].max(),
            text=f"Signal: {market_signal.upper()} ({signal_strength:.2f})",
            showarrow=False,
            font=dict(color=signal_color, size=10),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=signal_color,
            borderwidth=1,
            borderpad=2,
            xanchor="left",
            yanchor="top",
            row=i, col=1
        )
    
    # Update layout
    symbol = validation_data_by_timeframe[list(timeframes)[0]].get('symbol', '')
    option_type = validation_data_by_timeframe[list(timeframes)[0]].get('option_type', '')
    
    fig.update_layout(
        title=f"{symbol} {option_type} Option - Multi-Timeframe Analysis",
        xaxis_title="Time",
        height=200 * len(timeframes) + 100,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_sentiment_chart(sentiment_data):
    """
    Create a chart visualizing sentiment data
    
    Args:
        sentiment_data (dict): Sentiment data from recommendation
        
    Returns:
        dict: Plotly figure object
    """
    if not sentiment_data:
        return create_empty_chart("No sentiment data available")
    
    # Extract sentiment score
    score = sentiment_data.get('score', 0)
    label = sentiment_data.get('label', 'Neutral')
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Analysis", 'font': {'size': 24}},
        delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.5], 'color': 'red'},
                {'range': [-0.5, -0.2], 'color': 'salmon'},
                {'range': [-0.2, 0.2], 'color': 'lightgray'},
                {'range': [0.2, 0.5], 'color': 'lightgreen'},
                {'range': [0.5, 1], 'color': 'green'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    # Add annotation for sentiment label
    fig.add_annotation(
        x=0.5,
        y=0.25,
        text=f"Sentiment: {label}",
        showarrow=False,
        font=dict(size=16),
        xref="paper",
        yref="paper"
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_empty_chart(message="No data available"):
    """
    Create an empty chart with a message
    
    Args:
        message (str): Message to display
        
    Returns:
        dict: Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=message,
        showarrow=False,
        font=dict(size=16),
        xref="paper",
        yref="paper"
    )
    
    fig.update_layout(
        height=400,
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def create_risk_reward_visualization(recommendation):
    """
    Create a visualization of the risk/reward profile
    
    Args:
        recommendation (dict): Recommendation data
        
    Returns:
        dict: Plotly figure object
    """
    if not recommendation:
        return create_empty_chart("No recommendation data available")
    
    # Extract data
    symbol = recommendation.get('symbol', '')
    option_type = recommendation.get('option_type', 'CALL')
    strike = recommendation.get('strike', 0)
    entry_price = recommendation.get('entry_price', 0)
    current_price = recommendation.get('current_price', 0)
    target_price = recommendation.get('target_price', 0)
    max_loss = recommendation.get('max_loss', 0)
    potential_profit = recommendation.get('potential_profit', 0)
    risk_reward = recommendation.get('risk_reward_ratio', 0)
    
    # Create price range for x-axis
    if option_type == 'CALL':
        price_range = np.linspace(current_price * 0.9, target_price * 1.1, 100)
    else:  # PUT
        price_range = np.linspace(target_price * 0.9, current_price * 1.1, 100)
    
    # Calculate payoff at expiration
    payoff = []
    for price in price_range:
        if option_type == 'CALL':
            payoff_at_price = max(0, price - strike) - entry_price
        else:  # PUT
            payoff_at_price = max(0, strike - price) - entry_price
        payoff.append(payoff_at_price)
    
    # Create figure
    fig = go.Figure()
    
    # Add payoff line
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=payoff,
            mode='lines',
            name='Payoff at Expiration',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add breakeven point
    breakeven = strike + entry_price if option_type == 'CALL' else strike - entry_price
    fig.add_vline(
        x=breakeven, 
        line_dash="dash", 
        line_color="gray", 
        annotation_text="Breakeven",
        annotation_position="top"
    )
    
    # Add current price
    fig.add_vline(
        x=current_price, 
        line_dash="solid", 
        line_color="blue", 
        annotation_text="Current",
        annotation_position="top"
    )
    
    # Add strike price
    fig.add_vline(
        x=strike, 
        line_dash="dash", 
        line_color="purple", 
        annotation_text="Strike",
        annotation_position="bottom"
    )
    
    # Add target price
    fig.add_vline(
        x=target_price, 
        line_dash="dot", 
        line_color="green", 
        annotation_text="Target",
        annotation_position="top"
    )
    
    # Add zero line
    fig.add_hline(
        y=0, 
        line_dash="solid", 
        line_color="black"
    )
    
    # Add max loss and potential profit lines
    fig.add_hline(
        y=-max_loss, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Max Loss",
        annotation_position="left"
    )
    
    fig.add_hline(
        y=potential_profit, 
        line_dash="dash", 
        line_color="green", 
        annotation_text="Potential Profit",
        annotation_position="right"
    )
    
    # Add risk/reward annotation
    fig.add_annotation(
        x=price_range[len(price_range)//2],
        y=potential_profit/2,
        text=f"Risk/Reward: {risk_reward:.2f}x",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} {option_type} Option - Risk/Reward Profile",
        xaxis_title="Underlying Price at Expiration",
        yaxis_title="Profit/Loss",
        height=400,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig
