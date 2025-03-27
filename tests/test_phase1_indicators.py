"""
Test script for the newly implemented technical indicators in Phase 1.
This script creates sample data and tests each new indicator.
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
    volume = volume * (1 + 0.5 * np.abs(close.pct_change()))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df

def test_cmo():
    """Test Chande Momentum Oscillator implementation"""
    print("\n=== Testing Chande Momentum Oscillator ===")
    data = generate_sample_data()
    
    # Calculate CMO
    ti = TechnicalIndicators(data)
    cmo = ti.chande_momentum_oscillator()
    
    # Basic validation
    print(f"CMO data points: {len(cmo)}")
    print(f"CMO range: {cmo.min():.2f} to {cmo.max():.2f}")
    print(f"CMO first 5 values: {cmo.head().values}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title('Price')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(cmo.index, cmo)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.axhline(y=50, color='g', linestyle='--')
    plt.axhline(y=-50, color='g', linestyle='--')
    plt.title('Chande Momentum Oscillator')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cmo_test.png')
    print("CMO plot saved as cmo_test.png")
    
    return cmo

def test_stochastic_rsi():
    """Test Stochastic RSI implementation"""
    print("\n=== Testing Stochastic RSI ===")
    data = generate_sample_data()
    
    # Calculate Stochastic RSI
    ti = TechnicalIndicators(data)
    stoch_rsi = ti.stochastic_rsi()
    
    # Basic validation
    print(f"Stochastic RSI data points: {len(stoch_rsi)}")
    print(f"K range: {stoch_rsi['k'].min():.2f} to {stoch_rsi['k'].max():.2f}")
    print(f"D range: {stoch_rsi['d'].min():.2f} to {stoch_rsi['d'].max():.2f}")
    print(f"First 5 K values: {stoch_rsi['k'].head().values}")
    print(f"First 5 D values: {stoch_rsi['d'].head().values}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title('Price')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(stoch_rsi.index, stoch_rsi['k'], label='%K')
    plt.plot(stoch_rsi.index, stoch_rsi['d'], label='%D')
    plt.axhline(y=80, color='r', linestyle='--')
    plt.axhline(y=20, color='g', linestyle='--')
    plt.title('Stochastic RSI')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('stoch_rsi_test.png')
    print("Stochastic RSI plot saved as stoch_rsi_test.png")
    
    return stoch_rsi

def test_obv():
    """Test On-Balance Volume implementation"""
    print("\n=== Testing On-Balance Volume ===")
    data = generate_sample_data()
    
    # Calculate OBV
    ti = TechnicalIndicators(data)
    obv = ti.on_balance_volume()
    
    # Basic validation
    print(f"OBV data points: {len(obv)}")
    print(f"OBV range: {obv.min():.2f} to {obv.max():.2f}")
    print(f"OBV first 5 values: {obv.head().values}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title('Price')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(obv.index, obv)
    plt.title('On-Balance Volume')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('obv_test.png')
    print("OBV plot saved as obv_test.png")
    
    return obv

def test_adl():
    """Test Accumulation/Distribution Line implementation"""
    print("\n=== Testing Accumulation/Distribution Line ===")
    data = generate_sample_data()
    
    # Calculate A/D Line
    ti = TechnicalIndicators(data)
    adl = ti.accumulation_distribution_line()
    
    # Basic validation
    print(f"A/D Line data points: {len(adl)}")
    print(f"A/D Line range: {adl.min():.2f} to {adl.max():.2f}")
    print(f"A/D Line first 5 values: {adl.head().values}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title('Price')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(adl.index, adl)
    plt.title('Accumulation/Distribution Line')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('adl_test.png')
    print("A/D Line plot saved as adl_test.png")
    
    return adl

def test_ama():
    """Test Adaptive Moving Average implementation"""
    print("\n=== Testing Adaptive Moving Average ===")
    data = generate_sample_data()
    
    # Calculate AMA
    ti = TechnicalIndicators(data)
    ama = ti.adaptive_moving_average()
    
    # Basic validation
    print(f"AMA data points: {len(ama)}")
    print(f"AMA range: {ama.min():.2f} to {ama.max():.2f}")
    print(f"AMA first 5 values: {ama.head().values}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'], label='Price')
    plt.plot(ama.index, ama, label='AMA')
    plt.title('Price vs Adaptive Moving Average')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ama_test.png')
    print("AMA plot saved as ama_test.png")
    
    return ama

def test_volatility_regime():
    """Test Volatility Regime Identification implementation"""
    print("\n=== Testing Volatility Regime Identification ===")
    data = generate_sample_data(days=500)  # Need more data for meaningful regimes
    
    # Calculate Volatility Regime
    ti = TechnicalIndicators(data)
    regime = ti.volatility_regime()
    
    # Basic validation
    print(f"Regime data points: {len(regime)}")
    print(f"Regime value counts: {regime.value_counts()}")
    print(f"Regime first 5 values: {regime.head().values}")
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Price subplot
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title('Price')
    plt.grid(True)
    
    # Regime subplot
    plt.subplot(2, 1, 2)
    
    # Convert regime to numeric for plotting
    regime_numeric = pd.Series(index=regime.index)
    regime_numeric[regime == 'high'] = 1
    regime_numeric[regime == 'normal'] = 0
    regime_numeric[regime == 'low'] = -1
    
    # Plot as step function
    plt.step(regime_numeric.index, regime_numeric, where='mid')
    plt.yticks([-1, 0, 1], ['Low', 'Normal', 'High'])
    plt.title('Volatility Regime')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('volatility_regime_test.png')
    print("Volatility Regime plot saved as volatility_regime_test.png")
    
    return regime

def run_all_tests():
    """Run all indicator tests"""
    print("Starting technical indicator tests...")
    
    test_cmo()
    test_stochastic_rsi()
    test_obv()
    test_adl()
    test_ama()
    test_volatility_regime()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
