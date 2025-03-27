"""
Test script for Phase 2 multi-timeframe improvements.
Tests dynamic timeframe weighting, adaptive lookback periods, and timeframe confluence analysis.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.indicators.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from app.data_collector import DataCollector

class MockDataCollector:
    """
    Mock data collector for testing that doesn't require API access
    """
    def __init__(self, test_data_dir='tests/data'):
        self.test_data_dir = test_data_dir
        os.makedirs(test_data_dir, exist_ok=True)
        self.generate_test_data()
    
    def generate_test_data(self):
        """Generate synthetic test data for different timeframes"""
        # Base parameters for data generation
        start_date = datetime.now() - timedelta(days=60)
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        timeframes = {
            '1m': {'periods': 390, 'freq': 'T'},  # 1 day of minute data
            '5m': {'periods': 390, 'freq': '5T'},  # 5 days of 5-minute data
            '15m': {'periods': 260, 'freq': '15T'},  # 10 days of 15-minute data
            '1h': {'periods': 120, 'freq': 'H'},  # 20 days of hourly data
            '4h': {'periods': 90, 'freq': '4H'},  # 30 days of 4-hour data
            '1d': {'periods': 60, 'freq': 'D'}   # 60 days of daily data
        }
        
        # Generate data for each symbol and timeframe
        for symbol in symbols:
            for tf, params in timeframes.items():
                # Create date range
                date_range = pd.date_range(
                    start=start_date, 
                    periods=params['periods'], 
                    freq=params['freq']
                )
                
                # Generate price data with some randomness but realistic patterns
                np.random.seed(42)  # For reproducibility
                
                # Base price and trend
                base_price = 100.0
                trend = np.linspace(0, 20, params['periods'])  # Upward trend
                
                # Add some cyclicality
                cycles = 10 * np.sin(np.linspace(0, 4*np.pi, params['periods']))
                
                # Add random noise
                noise = np.random.normal(0, 1, params['periods'])
                
                # Combine components
                close_prices = base_price + trend + cycles + noise
                
                # Generate OHLC data
                high_prices = close_prices + np.random.uniform(0.5, 2.0, params['periods'])
                low_prices = close_prices - np.random.uniform(0.5, 2.0, params['periods'])
                open_prices = low_prices + np.random.uniform(0, 1, params['periods']) * (high_prices - low_prices)
                
                # Generate volume data
                volume = np.random.uniform(1000, 10000, params['periods'])
                # Make volume correlate somewhat with price changes
                price_changes = np.diff(close_prices, prepend=close_prices[0])
                volume = volume * (1 + 0.5 * np.abs(price_changes) / np.mean(np.abs(price_changes)))
                
                # Create DataFrame
                df = pd.DataFrame({
                    'open': open_prices,
                    'high': high_prices,
                    'low': low_prices,
                    'close': close_prices,
                    'volume': volume.astype(int)
                }, index=date_range)
                
                # Save to CSV
                os.makedirs(f"{self.test_data_dir}/{symbol}", exist_ok=True)
                df.to_csv(f"{self.test_data_dir}/{symbol}/{tf}.csv")
                
                print(f"Generated test data for {symbol} {tf}: {len(df)} rows")
    
    def get_historical_data(self, symbol, period_type='day', period=10, frequency_type='minute', frequency=1):
        """
        Mock implementation of get_historical_data that returns test data
        """
        # Map API parameters to our test data timeframes
        if frequency_type == 'minute':
            if frequency == 1:
                tf = '1m'
            elif frequency == 5:
                tf = '5m'
            elif frequency == 15:
                tf = '15m'
            else:
                tf = '1m'  # Default
        elif frequency_type == 'hour':
            if frequency == 1:
                tf = '1h'
            elif frequency == 4:
                tf = '4h'
            else:
                tf = '1h'  # Default
        elif frequency_type in ['daily', 'day']:
            tf = '1d'
        else:
            tf = '1d'  # Default
        
        # Load test data
        try:
            df = pd.read_csv(f"{self.test_data_dir}/{symbol}/{tf}.csv", index_col=0, parse_dates=True)
            return df
        except FileNotFoundError:
            print(f"Test data not found for {symbol} {tf}")
            return pd.DataFrame()

def test_dynamic_timeframe_weighting():
    """Test dynamic timeframe weighting based on market conditions"""
    print("\n=== Testing Dynamic Timeframe Weighting ===")
    
    # Initialize mock data collector and analyzer
    data_collector = MockDataCollector()
    analyzer = MultiTimeframeAnalyzer(data_collector)
    
    # Test with different market regimes
    test_regimes = [
        ('normal', 'moderate'),
        ('high', 'strong'),
        ('low', 'weak')
    ]
    
    for volatility, trend in test_regimes:
        # Set market regime manually for testing
        analyzer.current_volatility_regime = volatility
        analyzer.trend_strength = trend
        
        # Update weights based on regime
        analyzer._update_dynamic_weights()
        
        # Print results
        print(f"\nMarket Regime: Volatility={volatility}, Trend={trend}")
        print("Timeframe Weights:")
        for tf, weight in analyzer.timeframe_weights.items():
            print(f"  {tf}: {weight:.3f}")
    
    return True

def test_adaptive_lookback_periods():
    """Test adaptive lookback periods for different market regimes"""
    print("\n=== Testing Adaptive Lookback Periods ===")
    
    # Initialize mock data collector and analyzer
    data_collector = MockDataCollector()
    analyzer = MultiTimeframeAnalyzer(data_collector)
    
    # Test with different market regimes
    test_regimes = [
        ('normal', 'moderate'),
        ('high', 'strong'),
        ('low', 'weak')
    ]
    
    base_lookback = 30
    
    for volatility, trend in test_regimes:
        # Set market regime manually for testing
        analyzer.current_volatility_regime = volatility
        analyzer.trend_strength = trend
        
        # Calculate adaptive lookback
        adaptive_lookback = analyzer._calculate_adaptive_lookback(base_lookback)
        
        # Print results
        print(f"\nMarket Regime: Volatility={volatility}, Trend={trend}")
        print("Adaptive Lookback Periods:")
        for purpose, days in adaptive_lookback.items():
            print(f"  {purpose}: {days} days")
        
        # Test timeframe-specific lookback
        print("\nTimeframe-specific periods:")
        for tf in analyzer.timeframes:
            period = analyzer._get_adaptive_period_for_timeframe(tf, adaptive_lookback)
            print(f"  {tf}: {period} days")
    
    return True

def test_timeframe_confluence_analysis():
    """Test timeframe confluence analysis to identify stronger signals"""
    print("\n=== Testing Timeframe Confluence Analysis ===")
    
    # Initialize mock data collector and analyzer
    data_collector = MockDataCollector()
    analyzer = MultiTimeframeAnalyzer(data_collector)
    
    # Test with sample signal counts
    test_signal_counts = {
        'rsi': {'bullish': 3, 'bearish': 1, 'neutral': 2, 'total': 6},
        'macd': {'bullish': 4, 'bearish': 1, 'neutral': 1, 'total': 6},
        'bb': {'bullish': 2, 'bearish': 2, 'neutral': 2, 'total': 6},
        'cmo': {'bullish': 3, 'bearish': 2, 'neutral': 1, 'total': 6},
        'stoch_rsi': {'bullish': 4, 'bearish': 0, 'neutral': 2, 'total': 6},
        'volume': {'bullish': 3, 'bearish': 1, 'neutral': 2, 'total': 6},
        'trend': {'bullish': 5, 'bearish': 0, 'neutral': 1, 'total': 6},
        'candlestick': {'bullish': 2, 'bearish': 3, 'neutral': 1, 'total': 6}
    }
    
    # Analyze confluence
    confluence_results = analyzer._analyze_signal_confluence(test_signal_counts)
    
    # Print results
    print("\nConfluence Analysis Results:")
    print(f"Overall Bullish Confluence: {confluence_results['bullish_confluence']:.2f}")
    print(f"Overall Bearish Confluence: {confluence_results['bearish_confluence']:.2f}")
    print(f"Overall Neutral Confluence: {confluence_results['neutral_confluence']:.2f}")
    print(f"Strongest Bullish Indicator: {confluence_results['strongest_bullish_indicator']} ({confluence_results['strongest_bullish_percentage']:.2f})")
    print(f"Strongest Bearish Indicator: {confluence_results['strongest_bearish_indicator']} ({confluence_results['strongest_bearish_percentage']:.2f})")
    
    print("\nConfluence by Indicator:")
    for indicator in test_signal_counts.keys():
        bullish = confluence_results['bullish_by_indicator'].get(indicator, 0)
        bearish = confluence_results['bearish_by_indicator'].get(indicator, 0)
        neutral = confluence_results['neutral_by_indicator'].get(indicator, 0)
        print(f"  {indicator}: Bullish={bullish:.2f}, Bearish={bearish:.2f}, Neutral={neutral:.2f}")
    
    return True

def test_full_analysis():
    """Test full multi-timeframe analysis with all Phase 2 improvements"""
    print("\n=== Testing Full Multi-Timeframe Analysis ===")
    
    # Initialize mock data collector and analyzer
    data_collector = MockDataCollector()
    analyzer = MultiTimeframeAnalyzer(data_collector)
    
    # Test with a sample symbol
    symbol = 'AAPL'
    
    # Run full analysis
    results = analyzer.analyze_multi_timeframe(symbol, lookback_days=30)
    
    # Print summary results
    print(f"\nAnalysis Results for {symbol}:")
    print(f"Market Regime: Volatility={results['combined_signals']['market_regime']['volatility']}, Trend={results['combined_signals']['market_regime']['trend_strength']}")
    print(f"Overall Sentiment: {results['combined_signals']['overall_sentiment']}")
    print(f"Confidence: {results['combined_signals']['confidence']:.2f}")
    
    print("\nSignal Strengths:")
    print(f"  Bullish: {results['combined_signals']['bullish']:.2f}")
    print(f"  Bearish: {results['combined_signals']['bearish']:.2f}")
    print(f"  Neutral: {results['combined_signals']['neutral']:.2f}")
    
    print("\nDynamic Timeframe Weights:")
    for tf, weight in results['combined_signals']['dynamic_weights'].items():
        print(f"  {tf}: {weight:.3f}")
    
    print("\nConfluence Analysis:")
    confluence = results['combined_signals']['confluence_analysis']
    print(f"  Bullish Confluence: {confluence['bullish_confluence']:.2f}")
    print(f"  Bearish Confluence: {confluence['bearish_confluence']:.2f}")
    print(f"  Strongest Bullish: {confluence['strongest_bullish_indicator']} ({confluence['strongest_bullish_percentage']:.2f})")
    print(f"  Strongest Bearish: {confluence['strongest_bearish_indicator']} ({confluence['strongest_bearish_percentage']:.2f})")
    
    print("\nSignal Details:")
    for detail in results['combined_signals']['signal_details'][:10]:  # Show first 10 details
        print(f"  {detail}")
    
    # Plot timeframe weights
    plt.figure(figsize=(10, 6))
    timeframes = list(results['combined_signals']['dynamic_weights'].keys())
    weights = list(results['combined_signals']['dynamic_weights'].values())
    plt.bar(timeframes, weights)
    plt.title(f'Dynamic Timeframe Weights ({symbol})')
    plt.xlabel('Timeframe')
    plt.ylabel('Weight')
    plt.savefig('tests/output/dynamic_weights.png')
    
    # Plot signal strengths
    plt.figure(figsize=(10, 6))
    signals = ['Bullish', 'Bearish', 'Neutral']
    strengths = [
        results['combined_signals']['bullish'],
        results['combined_signals']['bearish'],
        results['combined_signals']['neutral']
    ]
    plt.bar(signals, strengths)
    plt.title(f'Signal Strengths ({symbol})')
    plt.xlabel('Signal Type')
    plt.ylabel('Strength')
    plt.savefig('tests/output/signal_strengths.png')
    
    return True

def run_all_tests():
    """Run all Phase 2 tests"""
    print("=== Running All Phase 2 Tests ===")
    
    tests = [
        test_dynamic_timeframe_weighting,
        test_adaptive_lookback_periods,
        test_timeframe_confluence_analysis,
        test_full_analysis
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            result = test()
            if not result:
                all_passed = False
                print(f"Test {test.__name__} failed")
            else:
                print(f"Test {test.__name__} passed")
        except Exception as e:
            all_passed = False
            print(f"Test {test.__name__} raised an exception: {str(e)}")
    
    if all_passed:
        print("\n=== All Phase 2 tests passed! ===")
    else:
        print("\n=== Some Phase 2 tests failed! ===")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
