"""
Test script for the TechnicalIndicators class with the new calculate_all_indicators method.
"""

import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('technical_indicators_test')

# Import the TechnicalIndicators class
sys.path.append('/home/ubuntu/workspace/options_copy_TB')
from app.indicators.technical_indicators import TechnicalIndicators

def generate_test_data(days=30, freq='1min'):
    """Generate sample price data for testing"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate random price data
    np.random.seed(42)  # For reproducibility
    
    # Initial price
    initial_price = 100.0
    
    # Generate random returns
    returns = np.random.normal(0, 0.01, len(date_range))
    
    # Calculate prices
    prices = initial_price * (1 + returns).cumprod()
    
    # Generate OHLCV data
    high = prices * (1 + np.random.uniform(0, 0.02, len(date_range)))
    low = prices * (1 - np.random.uniform(0, 0.02, len(date_range)))
    open_prices = prices * (1 + np.random.normal(0, 0.01, len(date_range)))
    close = prices
    volume = np.random.randint(1000, 10000, len(date_range))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_range)
    
    return df

def test_calculate_all_indicators():
    """Test the calculate_all_indicators method"""
    logger.info("Generating test data...")
    price_data = generate_test_data(days=30, freq='1min')
    logger.info(f"Generated price data with shape: {price_data.shape}")
    
    # Create TechnicalIndicators instance
    logger.info("Creating TechnicalIndicators instance...")
    ti = TechnicalIndicators()
    
    # Test calculate_all_indicators method
    logger.info("Calculating all indicators...")
    try:
        indicators = ti.calculate_all_indicators(price_data)
        logger.info(f"Successfully calculated indicators with shape: {indicators.shape}")
        
        # Check if key indicators are present
        expected_indicators = [
            'sma_20', 'ema_12', 'rsi_14', 'macd_line', 'macd_signal', 
            'bb_upper', 'bb_lower', 'bb_middle'
        ]
        
        missing_indicators = [ind for ind in expected_indicators if ind not in indicators.columns]
        
        if missing_indicators:
            logger.warning(f"Missing indicators: {missing_indicators}")
        else:
            logger.info("All expected indicators are present")
        
        # Print first few rows of indicators
        logger.info("Sample of calculated indicators:")
        logger.info(indicators.iloc[-5:].head())
        
        return True, indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return False, None

if __name__ == "__main__":
    logger.info("Starting technical indicators test...")
    success, indicators = test_calculate_all_indicators()
    logger.info(f"Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
