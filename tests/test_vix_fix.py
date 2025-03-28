"""
Test script for VIX symbol fix in options_collector.py
"""

import logging
import sys
import os

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data.options_collector import OptionsDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MockClient:
    def __init__(self):
        self.calls = []
        self.options_chain_calls = []
    
    def get_quote(self, symbol):
        self.calls.append(symbol)
        if symbol == 'VIX':
            # Return None for VIX to simulate failure
            return None
        elif symbol == '^VIX':
            # Return data for ^VIX to simulate success with caret prefix
            return {
                'openPrice': 15.0,
                'highPrice': 16.0,
                'lowPrice': 14.0,
                'lastPrice': 15.5,
                'totalVolume': 1000
            }
        else:
            # Return data for other symbols
            return {
                'openPrice': 100.0,
                'highPrice': 105.0,
                'lowPrice': 95.0,
                'lastPrice': 102.0,
                'totalVolume': 5000
            }
    
    def get_options_chain(self, symbol):
        self.options_chain_calls.append(symbol)
        # Return empty options chain for simplicity
        return {'callExpDateMap': {}, 'putExpDateMap': {}}

class MockDB:
    def __init__(self):
        self.stored_data = []
    
    def store_underlying_data(self, data):
        self.stored_data.extend(data)
        print(f'Stored data: {data}')
    
    def store_options_data(self, data):
        # Just a stub for the test
        pass

def test_vix_symbol_fix():
    print("Testing VIX symbol fix...")
    
    # Create mock objects
    mock_client = MockClient()
    mock_db = MockDB()
    
    # Create collector with SPY and VIX symbols
    collector = OptionsDataCollector(mock_client, mock_db, symbols=['SPY', 'VIX'])
    
    # Run collection
    collector._collect_data()
    
    # Check API calls
    print(f'API calls made: {mock_client.calls}')
    
    # Verify that both SPY and VIX were attempted
    assert 'SPY' in mock_client.calls, "SPY call not made"
    assert 'VIX' in mock_client.calls, "VIX call not made"
    
    # Verify that ^VIX was attempted after VIX failed
    assert '^VIX' in mock_client.calls, "^VIX call not made after VIX failed"
    
    # Verify that data was stored for both symbols
    symbols_stored = [item['symbol'] for item in mock_db.stored_data]
    print(f'Symbols stored: {symbols_stored}')
    
    assert 'SPY' in symbols_stored, "SPY data not stored"
    assert 'VIX' in symbols_stored, "VIX data not stored"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_vix_symbol_fix()
