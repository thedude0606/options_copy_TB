"""
Test script for the options recommendation system.
This script tests the integration of all components of the recommendation system.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
try:
    from app.data_collector import DataCollector
    from app.indicators.patterns.candlestick_patterns import CandlestickPatterns
    from app.indicators.multi_timeframe_analyzer import MultiTimeframeAnalyzer
    from app.analysis.profit_predictor import ProfitPredictor
    from app.analysis.confidence_calculator import ConfidenceCalculator
    from app.integration.data_integrator import DataIntegrator
    from app.components.enhanced_recommendation_display import EnhancedRecommendationDisplay
    
    logger.info("Successfully imported all modules")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def create_test_data():
    """
    Create test data for testing the recommendation system
    """
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
    
    # Create sample option data
    option_data = {
        'symbol': 'AAPL',
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
    
    # Create sample market data
    market_data = {
        'market_trend': 'bullish',
        'vix': 18.5,
        'sector_performance': 1.2,
        'indices': {
            'SPY': {'price': 450.0, 'change': 2.5, 'change_pct': 0.56},
            'QQQ': {'price': 380.0, 'change': 3.2, 'change_pct': 0.85},
            'DIA': {'price': 350.0, 'change': 1.8, 'change_pct': 0.52},
            'IWM': {'price': 220.0, 'change': 1.2, 'change_pct': 0.55}
        },
        'sectors': {
            'Technology': {'symbol': 'XLK', 'change_pct': 1.2},
            'Financial': {'symbol': 'XLF', 'change_pct': 0.8},
            'Energy': {'symbol': 'XLE', 'change_pct': -0.5},
            'Healthcare': {'symbol': 'XLV', 'change_pct': 0.3}
        }
    }
    
    return price_data, option_data, market_data

def test_candlestick_patterns():
    """
    Test the candlestick pattern recognition module
    """
    logger.info("Testing candlestick pattern recognition...")
    
    try:
        # Create test data
        price_data, _, _ = create_test_data()
        
        # Initialize candlestick patterns
        patterns = CandlestickPatterns()
        
        # Test pattern detection
        results = patterns.analyze_patterns(price_data)
        
        logger.info(f"Detected patterns: {results}")
        
        # Check if results are as expected
        assert isinstance(results, dict), "Results should be a dictionary"
        assert 'patterns' in results, "Results should contain 'patterns' key"
        
        logger.info("Candlestick pattern recognition test passed")
        return True
    except Exception as e:
        logger.error(f"Error testing candlestick patterns: {str(e)}")
        return False

def test_multi_timeframe_analyzer():
    """
    Test the multi-timeframe analyzer
    """
    logger.info("Testing multi-timeframe analyzer...")
    
    try:
        # Create test data
        price_data, _, _ = create_test_data()
        
        # Create timeframes
        timeframes = {
            '1d': price_data,
            '4h': price_data.iloc[-6:].reset_index(drop=True),  # Simulate 4h data
            '1h': price_data.iloc[-24:].reset_index(drop=True)  # Simulate 1h data
        }
        
        # Initialize candlestick patterns
        patterns = CandlestickPatterns()
        
        # Initialize multi-timeframe analyzer
        analyzer = MultiTimeframeAnalyzer(patterns)
        
        # Test multi-timeframe analysis
        results = analyzer.analyze_timeframes(timeframes)
        
        logger.info(f"Multi-timeframe analysis results: {results}")
        
        # Check if results are as expected
        assert isinstance(results, dict), "Results should be a dictionary"
        assert 'combined_signals' in results, "Results should contain 'combined_signals' key"
        
        logger.info("Multi-timeframe analyzer test passed")
        return True
    except Exception as e:
        logger.error(f"Error testing multi-timeframe analyzer: {str(e)}")
        return False

def test_profit_predictor():
    """
    Test the profit predictor
    """
    logger.info("Testing profit predictor...")
    
    try:
        # Create test data
        _, option_data, _ = create_test_data()
        
        # Initialize profit predictor
        predictor = ProfitPredictor()
        
        # Test profit prediction
        results = predictor.analyze_option_profit_potential(option_data)
        
        logger.info(f"Profit prediction results: {results}")
        
        # Check if results are as expected
        assert isinstance(results, dict), "Results should be a dictionary"
        assert 'profit_score' in results, "Results should contain 'profit_score' key"
        assert 'profit_probability' in results, "Results should contain 'profit_probability' key"
        assert 'exit_strategy' in results, "Results should contain 'exit_strategy' key"
        
        logger.info("Profit predictor test passed")
        return True
    except Exception as e:
        logger.error(f"Error testing profit predictor: {str(e)}")
        return False

def test_confidence_calculator():
    """
    Test the confidence calculator
    """
    logger.info("Testing confidence calculator...")
    
    try:
        # Create test data
        _, option_data, market_data = create_test_data()
        
        # Initialize multi-timeframe analyzer and profit predictor for the confidence calculator
        patterns = CandlestickPatterns()
        analyzer = MultiTimeframeAnalyzer(patterns)
        predictor = ProfitPredictor()
        
        # Initialize confidence calculator
        calculator = ConfidenceCalculator(analyzer, predictor)
        
        # Test confidence calculation
        results = calculator.calculate_confidence('AAPL', option_data, market_data)
        
        logger.info(f"Confidence calculation results: {results}")
        
        # Check if results are as expected
        assert isinstance(results, dict), "Results should be a dictionary"
        assert 'confidence_score' in results, "Results should contain 'confidence_score' key"
        assert 'confidence_level' in results, "Results should contain 'confidence_level' key"
        assert 'factors' in results, "Results should contain 'factors' key"
        
        logger.info("Confidence calculator test passed")
        return True
    except Exception as e:
        logger.error(f"Error testing confidence calculator: {str(e)}")
        return False

def test_data_integrator():
    """
    Test the data integrator
    """
    logger.info("Testing data integrator...")
    
    try:
        # Create mock data collector
        class MockDataCollector:
            def get_quote(self, symbol):
                return {
                    'lastPrice': 101.0,
                    'netChange': 1.5,
                    'percentChange': 1.5,
                    'totalVolume': 5000000
                }
            
            def get_option_chain(self, symbol, expiration_date=None):
                _, option_data, _ = create_test_data()
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
        
        # Initialize components
        patterns = CandlestickPatterns()
        analyzer = MultiTimeframeAnalyzer(patterns)
        predictor = ProfitPredictor()
        calculator = ConfidenceCalculator(analyzer, predictor)
        
        # Initialize data integrator with mock data collector
        integrator = DataIntegrator(MockDataCollector(), analyzer, predictor, calculator)
        
        # Test get market data
        market_data = integrator.get_market_data()
        logger.info(f"Market data: {market_data}")
        
        # Test get symbol market data
        symbol_data = integrator.get_symbol_market_data('AAPL')
        logger.info(f"Symbol market data: {symbol_data}")
        
        # Test analyze option
        _, option_data, _ = create_test_data()
        analysis = integrator.analyze_option('AAPL', option_data)
        logger.info(f"Option analysis: {analysis}")
        
        # Test get recommendations
        recommendations = integrator.get_recommendations('AAPL')
        logger.info(f"Recommendations: {recommendations}")
        
        # Check if results are as expected
        assert isinstance(recommendations, dict), "Recommendations should be a dictionary"
        assert 'symbol' in recommendations, "Recommendations should contain 'symbol' key"
        assert 'calls' in recommendations, "Recommendations should contain 'calls' key"
        assert 'puts' in recommendations, "Recommendations should contain 'puts' key"
        
        logger.info("Data integrator test passed")
        return True
    except Exception as e:
        logger.error(f"Error testing data integrator: {str(e)}")
        return False

def test_enhanced_recommendation_display():
    """
    Test the enhanced recommendation display
    """
    logger.info("Testing enhanced recommendation display...")
    
    try:
        # Create test data
        _, option_data, _ = create_test_data()
        
        # Add confidence data
        option_data['confidenceScore'] = 75
        option_data['confidenceLevel'] = 'high'
        option_data['recommendation'] = 'CALL recommendation. Good confidence with positive indicators.'
        option_data['signalDetails'] = [
            'Technical sentiment (bullish) aligns with CALL option: +0.60',
            'High delta (0.65) indicates high probability of being in-the-money: +0.80',
            'Moderate implied volatility (25.0%) is favorable for options trading: +0.60'
        ]
        option_data['expectedReturn'] = 25.5
        option_data['winRate'] = 0.65
        option_data['takeProfitPrice'] = 5.4
        option_data['stopLossPrice'] = 2.7
        option_data['optimalHoldDays'] = 15
        option_data['riskRewardRatio'] = 2.5
        
        # Initialize enhanced recommendation display
        display = EnhancedRecommendationDisplay()
        
        # Test create recommendation card
        card = display.create_recommendation_card(option_data, 'CALL')
        logger.info("Created recommendation card")
        
        # Test create recommendation detail modal
        modal = display.create_recommendation_detail_modal(option_data, 'CALL')
        logger.info("Created recommendation detail modal")
        
        # Test create profit projection chart
        chart = display.create_profit_projection_chart(option_data)
        logger.info("Created profit projection chart")
        
        # Test create recommendations grid
        recommendations = {
            'symbol': 'AAPL',
            'calls': [option_data],
            'puts': [{**option_data, 'optionType': 'PUT'}]
        }
        grid = display.create_recommendations_grid(recommendations)
        logger.info("Created recommendations grid")
        
        logger.info("Enhanced recommendation display test passed")
        return True
    except Exception as e:
        logger.error(f"Error testing enhanced recommendation display: {str(e)}")
        return False

def run_all_tests():
    """
    Run all tests
    """
    logger.info("Running all tests...")
    
    tests = [
        ("Candlestick Patterns", test_candlestick_patterns),
        ("Multi-Timeframe Analyzer", test_multi_timeframe_analyzer),
        ("Profit Predictor", test_profit_predictor),
        ("Confidence Calculator", test_confidence_calculator),
        ("Data Integrator", test_data_integrator),
        ("Enhanced Recommendation Display", test_enhanced_recommendation_display)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        success = test_func()
        results.append((name, success))
        logger.info(f"Test {name}: {'PASSED' if success else 'FAILED'}")
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    logger.info(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name}: {status}")
    
    return all(success for _, success in results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
