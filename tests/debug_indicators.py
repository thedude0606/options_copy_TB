"""
Test script for debugging the technical indicators that showed NaN values.
This script will help identify and fix issues with the indicators.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.indicators.technical_indicators import TechnicalIndicators

def generate_sample_data(days=200):
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with some trend and volatility
    close = np.random.normal(0, 1, size=len(dates))
    close = pd.Series(close).cumsum() + 100  # Start around 100 with random walk
    
    # Add some seasonality and trend
    t = np.arange(len(dates))
    close = close + 0.1 * t + 5 * np.sin(t / 20)
    
    # Generate other OHLC data
    high = close + np.random.uniform(0.5, 2, size=len(dates))
    low = close - np.random.uniform(0.5, 2, size=len(dates))
    open_price = low + np.random.uniform(0, 1, size=len(dates)) * (high - low)
    
    # Generate volume data
    volume = np.random.uniform(1000, 5000, size=len(dates))
    # Higher volume on bigger price moves
    volume = volume * (1 + 0.5 * np.abs(close.pct_change().fillna(0)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df

def debug_cmo():
    """Debug Chande Momentum Oscillator implementation"""
    print("\n=== Debugging Chande Momentum Oscillator ===")
    data = generate_sample_data()
    
    # Calculate price changes
    delta = data['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate sum of gains and losses over period
    period = 14
    sum_gain = gain.rolling(window=period).sum()
    sum_loss = loss.rolling(window=period).sum()
    
    # Check for zero denominators
    zero_denom = (sum_gain + sum_loss == 0)
    print(f"Zero denominators: {zero_denom.sum()}")
    
    # Calculate CMO with handling for zero denominators
    cmo = pd.Series(index=data.index)
    cmo[(sum_gain + sum_loss) > 0] = 100 * ((sum_gain - sum_loss) / (sum_gain + sum_loss))
    cmo[(sum_gain + sum_loss) == 0] = 0  # Handle zero denominator
    
    # Check for NaN values
    print(f"NaN values in CMO: {cmo.isna().sum()}")
    print(f"First 20 CMO values: {cmo.head(20).values}")
    
    return cmo

def debug_stoch_rsi():
    """Debug Stochastic RSI implementation"""
    print("\n=== Debugging Stochastic RSI ===")
    data = generate_sample_data()
    
    # Calculate RSI first
    period = 14
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Check RSI for NaN values
    print(f"NaN values in RSI: {rsi.isna().sum()}")
    
    # Apply Stochastic formula to RSI
    stoch_rsi = pd.Series(index=rsi.index)
    
    # Debug the stochastic calculation
    for i in range(period, len(rsi)):
        rsi_window = rsi.iloc[i-period+1:i+1]
        if not rsi_window.isna().all():  # Check if all values are NaN
            rsi_min = rsi_window.min()
            rsi_max = rsi_window.max()
            
            if not np.isnan(rsi_min) and not np.isnan(rsi_max) and (rsi_max - rsi_min) != 0:
                stoch_rsi.iloc[i] = (rsi.iloc[i] - rsi_min) / (rsi_max - rsi_min)
            else:
                stoch_rsi.iloc[i] = 0.5  # Default to middle if no range
    
    # Check for NaN values
    print(f"NaN values in Stochastic RSI: {stoch_rsi.isna().sum()}")
    print(f"First 30 Stochastic RSI values: {stoch_rsi.head(30).values}")
    
    return stoch_rsi

def debug_adl():
    """Debug Accumulation/Distribution Line implementation"""
    print("\n=== Debugging Accumulation/Distribution Line ===")
    data = generate_sample_data()
    
    # Calculate Money Flow Multiplier
    high_low_range = data['high'] - data['low']
    
    # Check for zero ranges
    zero_range = (high_low_range == 0)
    print(f"Zero high-low ranges: {zero_range.sum()}")
    
    # Avoid division by zero
    high_low_range = high_low_range.replace(0, np.nan)
    
    mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / high_low_range
    mfm = mfm.fillna(0)  # Replace NaN with 0
    
    # Calculate Money Flow Volume
    mfv = mfm * data['volume']
    
    # Calculate A/D Line (cumulative sum)
    adl = mfv.cumsum()
    
    # Check for NaN values
    print(f"NaN values in A/D Line: {adl.isna().sum()}")
    print(f"First 20 A/D Line values: {adl.head(20).values}")
    
    return adl

def debug_ama():
    """Debug Adaptive Moving Average implementation"""
    print("\n=== Debugging Adaptive Moving Average ===")
    data = generate_sample_data()
    
    er_period = 10
    
    # Calculate direction (absolute price change over er_period)
    direction = abs(data['close'].diff(er_period))
    
    # Calculate volatility (sum of absolute price changes over er_period)
    volatility = data['close'].diff().abs().rolling(window=er_period).sum()
    
    # Check for zero volatility
    zero_vol = (volatility == 0)
    print(f"Zero volatility: {zero_vol.sum()}")
    
    # Calculate Efficiency Ratio (ER) with handling for zero volatility
    er = pd.Series(index=data.index)
    er[volatility > 0] = direction[volatility > 0] / volatility[volatility > 0]
    er[volatility == 0] = 0  # Handle zero volatility
    
    # Calculate smoothing constant
    fast_period = 2
    slow_period = 30
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize AMA series
    ama = pd.Series(index=data.index)
    ama.iloc[0] = data['close'].iloc[0]
    
    # Calculate AMA
    for i in range(1, len(data)):
        if i < er_period:
            ama.iloc[i] = data['close'].iloc[i]  # Use price until we have enough data for ER
        else:
            ama.iloc[i] = ama.iloc[i-1] + sc.iloc[i] * (data['close'].iloc[i] - ama.iloc[i-1])
    
    # Check for NaN values
    print(f"NaN values in AMA: {ama.isna().sum()}")
    print(f"First 20 AMA values: {ama.head(20).values}")
    
    return ama

def run_debug_tests():
    """Run all debugging tests"""
    print("Starting debugging tests...")
    
    debug_cmo()
    debug_stoch_rsi()
    debug_adl()
    debug_ama()
    
    print("\nAll debugging tests completed!")

if __name__ == "__main__":
    run_debug_tests()
