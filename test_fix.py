"""
Test script to verify the fix for the DataCollector.get_historical_data method.
This script simulates the call path that was failing in the error trace.
"""
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_collector import DataCollector
from app.analysis.recommendation_engine import RecommendationEngine

def test_get_historical_data():
    """Test the get_historical_data method in DataCollector"""
    print("=== TESTING GET_HISTORICAL_DATA FIX ===")
    
    # Initialize DataCollector
    print("Initializing DataCollector...")
    data_collector = DataCollector(interactive_auth=False)
    
    # Test direct call to get_historical_data
    print("\nTesting direct call to get_historical_data...")
    try:
        historical_data = data_collector.get_historical_data(
            symbol="SPY",
            period_type="month",
            period=1,
            frequency_type="daily",
            frequency=1
        )
        print(f"Success! Retrieved {len(historical_data)} rows of historical data")
        print(f"Columns: {historical_data.columns.tolist() if not historical_data.empty else 'None'}")
    except Exception as e:
        print(f"Error calling get_historical_data directly: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # Test through RecommendationEngine
    print("\nTesting through RecommendationEngine...")
    try:
        # Initialize RecommendationEngine with our DataCollector
        recommendation_engine = RecommendationEngine(data_collector)
        
        # Generate recommendations which will call get_historical_data
        recommendations = recommendation_engine.generate_recommendations(
            symbol="SPY",
            lookback_days=30,
            confidence_threshold=0.6
        )
        
        print(f"Success! RecommendationEngine generated recommendations")
        print(f"Recommendations shape: {recommendations.shape if hasattr(recommendations, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_get_historical_data()
