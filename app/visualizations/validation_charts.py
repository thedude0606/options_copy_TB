"""
Validation visualizations for options recommendations.
Provides visual confirmation of technical patterns and trading signals.
"""
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_validation_chart(symbol, option_type, strike_price, expiration_date, timeframe="30m"):
    """
    Create a validation chart for a recommendation
    
    Args:
        symbol (str): Stock symbol
        option_type (str): Option type (call or put)
        strike_price (float): Strike price
        expiration_date (str): Expiration date
        timeframe (str, optional): Timeframe for analysis. Defaults to "30m".
        
    Returns:
        dict: Plotly figure object
    """
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    try:
        # Get historical data for the symbol
        from app.data_collector import DataCollector
        data_collector = DataCollector()
        
        # Get price data for the selected timeframe
        price_data = data_collector.get_historical_data(
            symbol=symbol,
            period_type="day",
            period_value=5,
            freq_type="minute",
            freq_value=int(timeframe.replace("m", ""))
        )
        
        if price_data is None or price_data.empty:
            # Return empty chart with message if no data
            fig.add_annotation(
                text="No historical data available for validation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
            
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
                name="Price"
            )
        )
        
        # Add strike price line
        fig.add_shape(
            type="line",
            x0=price_data['datetime'].iloc[0],
            y0=strike_price,
            x1=price_data['datetime'].iloc[-1],
            y1=strike_price,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        # Add annotation for strike price
        fig.add_annotation(
            x=price_data['datetime'].iloc[-1],
            y=strike_price,
            text=f"Strike: {strike_price}",
            showarrow=False,
            yshift=10,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        
        # Add volume bars on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=price_data['datetime'],
                y=price_data['volume'],
                name="Volume",
                marker_color='rgba(0, 0, 255, 0.3)'
            ),
            secondary_y=True
        )
        
        # Add expiration date line
        try:
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            fig.add_shape(
                type="line",
                x0=exp_date,
                y0=price_data['low'].min(),
                x1=exp_date,
                y1=price_data['high'].max(),
                line=dict(
                    color="green",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add annotation for expiration
            fig.add_annotation(
                x=exp_date,
                y=price_data['high'].max(),
                text=f"Expiration: {expiration_date}",
                showarrow=False,
                yshift=10,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        except Exception as e:
            print(f"Error adding expiration line: {str(e)}")
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} {option_type.upper()} ${strike_price} Validation ({timeframe})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True)
        
        return fig
        
    except Exception as e:
        import traceback
        print(f"Error creating validation chart: {str(e)}")
        print(traceback.format_exc())
        
        # Return empty chart with error message
        fig.add_annotation(
            text=f"Error creating validation chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_timeframe_comparison_chart(symbol, timeframes=["15m", "30m", "60m", "120m"]):
    """
    Create a comparison chart for different timeframes
    
    Args:
        symbol (str): Stock symbol
        timeframes (list): List of timeframes to compare
        
    Returns:
        dict: Plotly figure object
    """
    # Create subplot figure with one row per timeframe
    fig = make_subplots(
        rows=len(timeframes), 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"{symbol} - {tf} Timeframe" for tf in timeframes]
    )
    
    try:
        # Get historical data for the symbol
        from app.data_collector import DataCollector
        data_collector = DataCollector()
        
        for i, timeframe in enumerate(timeframes):
            # Get price data for the timeframe
            period_value = 1
            if timeframe == "60m":
                period_value = 2
            elif timeframe == "120m":
                period_value = 3
                
            price_data = data_collector.get_historical_data(
                symbol=symbol,
                period_type="day",
                period_value=period_value,
                freq_type="minute",
                freq_value=int(timeframe.replace("m", ""))
            )
            
            if price_data is None or price_data.empty:
                # Add annotation if no data
                fig.add_annotation(
                    text=f"No data available for {timeframe}",
                    xref="x",
                    yref="paper",
                    x=0.5,
                    y=0.5 - (i * 0.25),
                    showarrow=False
                )
                continue
                
            # Convert datetime to proper format if needed
            if 'datetime' in price_data.columns and not pd.api.types.is_datetime64_any_dtype(price_data['datetime']):
                price_data['datetime'] = pd.to_datetime(price_data['datetime'])
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_data['datetime'],
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'],
                    name=f"{timeframe}",
                    increasing_line_color='#26a69a', 
                    decreasing_line_color='#ef5350'
                ),
                row=i+1, col=1
            )
            
            # Add volume as bar chart
            fig.add_trace(
                go.Bar(
                    x=price_data['datetime'],
                    y=price_data['volume'],
                    name=f"Volume ({timeframe})",
                    marker_color='rgba(0, 0, 255, 0.3)',
                    opacity=0.7,
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=200 * len(timeframes),
            title=f"{symbol} - Timeframe Comparison",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        # Update y-axis labels
        for i in range(len(timeframes)):
            fig.update_yaxes(title_text="Price ($)", row=i+1, col=1)
        
        return fig
        
    except Exception as e:
        import traceback
        print(f"Error creating timeframe comparison chart: {str(e)}")
        print(traceback.format_exc())
        
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating timeframe comparison chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_sentiment_chart(symbol):
    """
    Create a sentiment analysis chart for a symbol
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Plotly figure object
    """
    fig = go.Figure()
    
    try:
        # Get historical data for the symbol
        from app.data_collector import DataCollector
        data_collector = DataCollector()
        
        # Get price data for the last 30 days
        price_data = data_collector.get_historical_data(
            symbol=symbol,
            period_type="month",
            period_value=1,
            freq_type="daily",
            freq_value=1
        )
        
        if price_data is None or price_data.empty:
            # Return empty chart with message if no data
            fig.add_annotation(
                text="No historical data available for sentiment analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
            
        # Convert datetime to proper format if needed
        if 'datetime' in price_data.columns and not pd.api.types.is_datetime64_any_dtype(price_data['datetime']):
            price_data['datetime'] = pd.to_datetime(price_data['datetime'])
        
        # Calculate simple sentiment indicators
        price_data['price_change'] = price_data['close'].pct_change()
        price_data['sentiment'] = price_data['price_change'].apply(
            lambda x: 'positive' if x > 0.01 else ('negative' if x < -0.01 else 'neutral')
        )
        
        # Count sentiment days
        sentiment_counts = price_data['sentiment'].value_counts()
        
        # Create pie chart
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker_colors=['#26a69a', '#b0bec5', '#ef5350'],
                textinfo='label+percent',
                hoverinfo='label+value'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - 30-Day Market Sentiment",
            annotations=[
                dict(
                    text=f"{symbol}",
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )
            ],
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        import traceback
        print(f"Error creating sentiment chart: {str(e)}")
        print(traceback.format_exc())
        
        # Return empty chart with error message
        fig.add_annotation(
            text=f"Error creating sentiment chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_risk_reward_visualization(symbol, option_type, strike_price, expiration_date):
    """
    Create a risk/reward visualization for an options trade
    
    Args:
        symbol (str): Stock symbol
        option_type (str): Option type (call or put)
        strike_price (float): Strike price
        expiration_date (str): Expiration date
        
    Returns:
        dict: Plotly figure object
    """
    fig = go.Figure()
    
    try:
        # Get historical data for the symbol
        from app.data_collector import DataCollector
        data_collector = DataCollector()
        
        # Get current price data
        price_data = data_collector.get_historical_data(
            symbol=symbol,
            period_type="day",
            period_value=1,
            freq_type="minute",
            freq_value=30
        )
        
        if price_data is None or price_data.empty:
            # Return empty chart with message if no data
            fig.add_annotation(
                text="No price data available for risk/reward analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get current price
        current_price = price_data['close'].iloc[-1]
        
        # Calculate price range for visualization
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        
        # Calculate option payoff at expiration
        if option_type.upper() == 'CALL':
            payoff = [max(0, price - strike_price) for price in price_range]
        else:  # PUT
            payoff = [max(0, strike_price - price) for price in price_range]
        
        # Estimate option premium (simplified)
        days_to_expiration = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days
        if days_to_expiration < 0:
            days_to_expiration = 1  # Minimum 1 day
            
        # Very simplified premium calculation
        volatility_factor = 0.2  # Assumed volatility
        time_factor = np.sqrt(days_to_expiration / 365)
        distance_factor = abs(current_price - strike_price) / current_price
        premium = current_price * volatility_factor * time_factor * (1 - distance_factor)
        premium = max(premium, 0.01 * current_price)  # Minimum premium
        
        # Calculate net payoff (payoff - premium)
        net_payoff = [p - premium for p in payoff]
        
        # Add breakeven line
        breakeven_price = strike_price + premium if option_type.upper() == 'CALL' else strike_price - premium
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=price_range,
                y=payoff,
                mode='lines',
                name='Option Payoff',
                line=dict(color='green' if option_type.upper() == 'CALL' else 'red', width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_range,
                y=net_payoff,
                mode='lines',
                name='Net Payoff (after premium)',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=price_range[0],
            y0=0,
            x1=price_range[-1],
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        # Add vertical lines for key prices
        fig.add_shape(
            type="line",
            x0=current_price,
            y0=min(min(net_payoff), -premium) * 1.1,
            x1=current_price,
            y1=max(payoff) * 1.1,
            line=dict(color="blue", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=strike_price,
            y0=min(min(net_payoff), -premium) * 1.1,
            x1=strike_price,
            y1=max(payoff) * 1.1,
            line=dict(color="purple", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=breakeven_price,
            y0=min(min(net_payoff), -premium) * 1.1,
            x1=breakeven_price,
            y1=max(payoff) * 1.1,
            line=dict(color="green", width=1, dash="dash")
        )
        
        # Add annotations
        fig.add_annotation(
            x=current_price,
            y=max(payoff) * 0.9,
            text=f"Current: ${current_price:.2f}",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        
        fig.add_annotation(
            x=strike_price,
            y=max(payoff) * 0.8,
            text=f"Strike: ${strike_price:.2f}",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        
        fig.add_annotation(
            x=breakeven_price,
            y=max(payoff) * 0.7,
            text=f"Breakeven: ${breakeven_price:.2f}",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        
        # Add premium annotation
        fig.add_annotation(
            x=price_range[0],
            y=-premium,
            text=f"Premium: ${premium:.2f}",
            showarrow=False,
            xanchor="left",
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} {option_type.upper()} ${strike_price} - Risk/Reward Profile",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        import traceback
        print(f"Error creating risk/reward visualization: {str(e)}")
        print(traceback.format_exc())
        
        # Return empty chart with error message
        fig.add_annotation(
            text=f"Error creating risk/reward visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
