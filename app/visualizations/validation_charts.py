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
