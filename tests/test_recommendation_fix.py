"""
Test script to verify that the DataFrame boolean evaluation fixes resolved the error.
"""
import sys
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
try:
    from app.integration.data_integrator import DataIntegrator
    from app.indicators.patterns.candlestick_patterns import CandlestickPatterns
    from app.indicators.multi_timeframe_analyzer import MultiTimeframeAnalyzer
    from app.analysis.profit_predictor import ProfitPredictor
    from app.analysis.confidence_calculator import ConfidenceCalculator
    
    print("Successfully imported all modules")
except Exception as e:
    print(f"Error importing modules: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

def create_mock_data_collector():
    """Create a mock data collector for testing"""
    class MockDataCollector:
        def get_quote(self, symbol):
            return {
                'lastPrice': 101.0,
                'netChange': 1.5,
                'percentChange': 1.5,
                'totalVolume': 5000000
            }
        
        def get_option_chain(self, symbol, expiration_date=None):
            option_data = {
                'symbol': symbol,
                'optionType': 'CALL',
                'strikePrice': 100.0,
                'expirationDate': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'daysToExpiration': 30,
                'bid': 3.5,
                'ask': 3.7,
                'mark': 3.6,
                'entryPrice': 3.6,
                'delta': 0.65,
                'gamma': 0.03,
                'theta': -0.05,
                'vega': 0.12,
                'rho': 0.01,
                'impliedVolatility': 0.25,
                'totalVolume': 1500,
                'openInterest': 5000,
                'underlyingPrice': 101.0
            }
            
            return {
                'symbol': symbol,
                'callExpDateMap': {
                    '2025-04-30:30': {
                        '100.0': [option_data]
                    }
                },
                'putExpDateMap': {
                    '2025-04-30:30': {
                        '100.0': [{**option_data, 'optionType': 'PUT'}]
                    }
                }
            }
        
        def get_price_history(self, symbol, **kwargs):
            # Create sample price data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            price_data = pd.DataFrame({
                'date': dates,
                'open': np.random.normal(100, 2, len(dates)),
                'high': np.random.normal(102, 2, len(dates)),
                'low': np.random.normal(98, 2, len(dates)),
                'close': np.random.normal(101, 2, len(dates)),
                'volume': np.random.randint(1000, 10000, len(dates))
            })
            
            # Ensure high is highest, low is lowest
            for i in range(len(price_data)):
                values = [price_data.loc[i, 'open'], price_data.loc[i, 'close']]
                price_data.loc[i, 'high'] = max(values) + abs(np.random.normal(0, 0.5))
                price_data.loc[i, 'low'] = min(values) - abs(np.random.normal(0, 0.5))
                
            return price_data
            
        # Add the get_historical_data method that was missing
        def get_historical_data(self, symbol, **kwargs):
            return self.get_price_history(symbol, **kwargs)
    
    return MockDataCollector()

def test_recommendation_generation():
    """Test the recommendation generation to verify the DataFrame evaluation fix"""
    print("\n=== Testing Recommendation Generation After Fix ===")
    
    try:
        # Create mock components
        data_collector = create_mock_data_collector()
        patterns = CandlestickPatterns()
        analyzer = MultiTimeframeAnalyzer(data_collector)
        predictor = ProfitPredictor()
        calculator = ConfidenceCalculator(analyzer, predictor)
        
        # Create DataIntegrator
        integrator = DataIntegrator(data_collector, analyzer, predictor, calculator)
        
        # Test get_recommendations
        print("\nTesting get_recommendations()...")
        try:
            recommendations = integrator.get_recommendations('AAPL')
            print("✓ get_recommendations() successful")
            print(f"Recommendations shape: {recommendations.shape if hasattr(recommendations, 'shape') else 'Not a DataFrame'}")
            print(f"Recommendations type: {type(recommendations)}")
            if hasattr(recommendations, 'columns'):
                print(f"Recommendations columns: {recommendations.columns.tolist()}")
            return True
        except Exception as e:
            print(f"✗ get_recommendations() failed: {str(e)}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test recommendation generation
    success = test_recommendation_generation()
    
    if success:
        print("\n✅ DataFrame evaluation fix was successful! The error has been resolved.")
        sys.exit(0)
    else:
        print("\n❌ DataFrame evaluation fix was not successful. The error still persists.")
        sys.exit(1)
