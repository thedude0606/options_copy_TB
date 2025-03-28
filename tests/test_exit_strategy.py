"""
Test script for exit strategy recommendations

This script tests the enhanced exit strategy recommendation functionality
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import recommendation engine
from app.analysis.exit_strategy_recommendation_engine import ExitStrategyEnhancedRecommendationEngine
from app.data_collector import DataCollector

def test_exit_strategy_recommendations():
    """Test exit strategy recommendations"""
    print("Testing exit strategy recommendations...")
    
    # Create data collector
    data_collector = DataCollector()
    
    # Create recommendation engine
    recommendation_engine = ExitStrategyEnhancedRecommendationEngine(
        data_collector=data_collector,
        debug=True
    )
    
    # Generate recommendations for SPY
    print("Generating recommendations for SPY...")
    recommendations = recommendation_engine.generate_recommendations(
        symbol="SPY",
        lookback_days=30,
        confidence_threshold=0.6,
        max_recommendations=5
    )
    
    # Check if recommendations were generated
    if recommendations.empty:
        print("No recommendations generated.")
        return
    
    print(f"Generated {len(recommendations)} recommendations.")
    
    # Check if exit strategy is included
    if 'exitStrategy' in recommendations.columns:
        print("Exit strategy column found in recommendations.")
    else:
        print("Exit strategy column NOT found in recommendations.")
        return
    
    # Print recommendation details
    for i, rec in recommendations.iterrows():
        print(f"\nRecommendation {i+1}:")
        print(f"Symbol: {rec.get('symbol', 'N/A')}")
        print(f"Strike: ${rec.get('strikePrice', 0):.2f}")
        print(f"Option Type: {rec.get('option_type', 'N/A')}")
        print(f"Expiration: {rec.get('expirationDate', 'N/A')}")
        print(f"Entry Price: ${rec.get('entryPrice', 0):.2f}")
        
        # Print exit strategy details
        exit_strategy = rec.get('exitStrategy', {})
        if exit_strategy:
            print("\nExit Strategy:")
            print(f"Optimal Exit Date: {exit_strategy.get('optimalExitDate', 'N/A')}")
            print(f"Days to Hold: {exit_strategy.get('daysToHold', 0)}")
            print(f"Take Profit: ${exit_strategy.get('takeProfit', 0):.2f}")
            print(f"Stop Loss: ${exit_strategy.get('stopLoss', 0):.2f}")
            print(f"Exit Probability: {exit_strategy.get('exitProbability', 0):.2f}")
            
            # Print price targets
            price_targets = exit_strategy.get('priceTargets', [])
            if price_targets:
                print("\nPrice Targets:")
                for i, target in enumerate(price_targets):
                    print(f"Target {i+1}: ${target.get('price', 0):.2f} "
                          f"({target.get('profit_percentage', 0):.1f}% profit, "
                          f"{target.get('percentage', 0)*100:.0f}% of position)")
        else:
            print("No exit strategy found for this recommendation.")
    
    print("\nTest completed.")

def test_exit_strategy_for_existing_position():
    """Test generating exit strategy for an existing position"""
    print("\nTesting exit strategy for existing position...")
    
    # Create data collector
    data_collector = DataCollector()
    
    # Create recommendation engine
    recommendation_engine = ExitStrategyEnhancedRecommendationEngine(
        data_collector=data_collector,
        debug=True
    )
    
    # Create sample position data
    position_data = {
        'symbol': 'SPY_20250418C450',
        'entry_price': 5.25,
        'entry_date': datetime.now() - timedelta(days=5),
        'position_type': 'long',
        'option_data': {
            'symbol': 'SPY_20250418C450',
            'underlying': 'SPY',
            'option_type': 'CALL',
            'strike': 450,
            'expiration_date': '2025-04-18',
            'daysToExpiration': 21,
            'implied_volatility': 0.25,
            'delta': 0.55,
            'gamma': 0.04,
            'theta': -0.03,
            'vega': 0.15,
            'rho': 0.01,
            'underlyingPrice': 452.75
        }
    }
    
    # Generate exit strategy
    print("Generating exit strategy for existing position...")
    exit_strategy = recommendation_engine.generate_exit_strategy_for_position(position_data)
    
    # Print exit strategy details
    if exit_strategy:
        print("\nExit Strategy:")
        print(f"Symbol: {exit_strategy.get('symbol', 'N/A')}")
        print(f"Optimal Exit Time: {exit_strategy.get('optimal_exit_time', 'N/A')}")
        print(f"Days to Hold: {exit_strategy.get('days_to_hold', 0)}")
        print(f"Take Profit: ${exit_strategy.get('take_profit', 0):.2f}")
        print(f"Stop Loss: ${exit_strategy.get('stop_loss', 0):.2f}")
        print(f"Exit Probability: {exit_strategy.get('exit_probability', 0):.2f}")
        
        # Print price targets
        price_targets = exit_strategy.get('price_targets', [])
        if price_targets:
            print("\nPrice Targets:")
            for i, target in enumerate(price_targets):
                print(f"Target {i+1}: ${target.get('price', 0):.2f} "
                      f"({target.get('profit_percentage', 0):.1f}% profit, "
                      f"{target.get('percentage', 0)*100:.0f}% of position)")
        
        # Print exit reasons
        exit_reasons = exit_strategy.get('exit_reasons', [])
        if exit_reasons:
            print("\nExit Reasons:")
            for reason in exit_reasons:
                print(f"- {reason}")
        
        # Print profit projections
        profit_projections = exit_strategy.get('profit_projections', {})
        if profit_projections:
            print("\nProfit Projections:")
            print(f"Expected Value: ${profit_projections.get('expected_value', 0):.2f}")
            print(f"Max Profit Potential: ${profit_projections.get('max_profit_potential', 0):.2f}")
            print(f"Max Loss Potential: ${profit_projections.get('max_loss_potential', 0):.2f}")
            print(f"Risk-Reward Ratio: {profit_projections.get('risk_reward_ratio', 0):.2f}x")
            
            # Print scenarios
            scenarios = profit_projections.get('scenarios', [])
            if scenarios:
                print("\nScenarios:")
                for scenario in scenarios:
                    print(f"- {scenario.get('description', 'N/A')}: "
                          f"${scenario.get('exit_price', 0):.2f} "
                          f"({scenario.get('profit_percentage', 0):.1f}% profit, "
                          f"{scenario.get('probability', 0)*100:.0f}% probability, "
                          f"{scenario.get('days_to_target', 0):.1f} days)")
    else:
        print("No exit strategy generated.")
    
    print("\nTest completed.")

def test_update_exit_strategy():
    """Test updating exit strategy with new data"""
    print("\nTesting update exit strategy with new data...")
    
    # Create data collector
    data_collector = DataCollector()
    
    # Create recommendation engine
    recommendation_engine = ExitStrategyEnhancedRecommendationEngine(
        data_collector=data_collector,
        debug=True
    )
    
    # Create sample position data
    position_data = {
        'symbol': 'SPY_20250418C450',
        'entry_price': 5.25,
        'entry_date': datetime.now() - timedelta(days=5),
        'position_type': 'long',
        'option_data': {
            'symbol': 'SPY_20250418C450',
            'underlying': 'SPY',
            'option_type': 'CALL',
            'strike': 450,
            'expiration_date': '2025-04-18',
            'daysToExpiration': 21,
            'implied_volatility': 0.25,
            'delta': 0.55,
            'gamma': 0.04,
            'theta': -0.03,
            'vega': 0.15,
            'rho': 0.01,
            'underlyingPrice': 452.75
        }
    }
    
    # Generate initial exit strategy
    print("Generating initial exit strategy...")
    exit_strategy = recommendation_engine.generate_exit_strategy_for_position(position_data)
    
    # Update exit strategy with new data
    print("Updating exit strategy with new data...")
    current_price = 6.50  # Option price has increased
    days_held = 5
    updated_strategy = recommendation_engine.update_exit_strategy_with_new_data(
        exit_strategy, current_price, days_held
    )
    
    # Print updated exit strategy details
    if updated_strategy:
        print("\nUpdated Exit Strategy:")
        print(f"Symbol: {updated_strategy.get('symbol', 'N/A')}")
        print(f"Current P&L: {updated_strategy.get('current_pnl_percentage', 0):.1f}%")
        print(f"Days Held So Far: {updated_strategy.get('days_held_so_far', 0)}")
        print(f"Days Remaining: {updated_strategy.get('days_remaining', 0)}")
        
        # Print update reason if available
        update_reason = updated_strategy.get('update_reason', [])
        if update_reason:
            print("\nUpdate Reason:")
            for reason in update_reason:
                print(f"- {reason}")
        
        # Print updated price targets if available
        price_targets = updated_strategy.get('price_targets', [])
        if price_targets:
            print("\nUpdated Price Targets:")
            for i, target in enumerate(price_targets):
                print(f"Target {i+1}: ${target.get('price', 0):.2f} "
                      f"({target.get('profit_percentage', 0):.1f}% profit, "
                      f"{target.get('percentage', 0)*100:.0f}% of position)")
    else:
        print("No updated exit strategy generated.")
    
    print("\nTest completed.")

if __name__ == "__main__":
    # Run tests
    test_exit_strategy_recommendations()
    test_exit_strategy_for_existing_position()
    test_update_exit_strategy()
