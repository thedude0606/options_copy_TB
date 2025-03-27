"""
Test module for candlestick pattern recognition.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.indicators.patterns.candlestick_patterns import CandlestickPatterns

def test_candlestick_patterns():
    """
    Test candlestick pattern detection with sample data
    """
    # Create sample data for testing
    data = pd.DataFrame({
        'open': [100, 105, 110, 100, 90, 95, 105, 110, 105, 100],
        'high': [110, 115, 115, 105, 95, 105, 115, 115, 110, 105],
        'low': [95, 100, 105, 85, 85, 90, 100, 100, 95, 90],
        'close': [105, 110, 100, 90, 92, 105, 110, 105, 100, 95]
    })
    
    # Test pattern detection
    results = []
    for idx in range(len(data)):
        patterns = CandlestickPatterns.analyze_candle(data, idx)
        if patterns:
            results.append({
                'index': idx,
                'candle': f"O:{data.iloc[idx]['open']} H:{data.iloc[idx]['high']} L:{data.iloc[idx]['low']} C:{data.iloc[idx]['close']}",
                'patterns': patterns
            })
    
    # Print results
    print("\n=== CANDLESTICK PATTERN TEST RESULTS ===")
    for result in results:
        print(f"Candle {result['index']}: {result['candle']}")
        for pattern, strength in result['patterns'].items():
            print(f"  - {pattern.replace('_', ' ').title()}: {strength:.2f}")
    
    # Test full analysis
    analysis_results = CandlestickPatterns.analyze_candlestick_patterns(data)
    
    print("\n=== FULL ANALYSIS RESULTS ===")
    for result in analysis_results:
        print(f"Candle {result['index']}: Sentiment={result['sentiment']}, Strength={result['strength']:.2f}")
        for pattern, strength in result['patterns'].items():
            print(f"  - {pattern.replace('_', ' ').title()}: {strength:.2f}")
    
    return len(results) > 0

if __name__ == "__main__":
    test_candlestick_patterns()
