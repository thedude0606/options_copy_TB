"""
Test script for Phase 3 scoring algorithm refinements.
Tests dynamic weighting system, confidence score calibration, and strategy-specific scoring models.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analysis.recommendation_engine_phase3 import RecommendationEngine
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
        """Generate synthetic test data for historical prices and options"""
        # Generate historical price data
        self._generate_historical_data()
        
        # Generate options data
        self._generate_options_data()
    
    def _generate_historical_data(self):
        """Generate synthetic historical price data"""
        # Base parameters for data generation
        start_date = datetime.now() - timedelta(days=60)
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            # Create date range
            date_range = pd.date_range(
                start=start_date, 
                periods=60, 
                freq='D'
            )
            
            # Generate price data with some randomness but realistic patterns
            np.random.seed(42)  # For reproducibility
            
            # Base price and trend
            base_price = 100.0
            trend = np.linspace(0, 20, 60)  # Upward trend
            
            # Add some cyclicality
            cycles = 10 * np.sin(np.linspace(0, 4*np.pi, 60))
            
            # Add random noise
            noise = np.random.normal(0, 1, 60)
            
            # Combine components
            close_prices = base_price + trend + cycles + noise
            
            # Generate OHLC data
            high_prices = close_prices + np.random.uniform(0.5, 2.0, 60)
            low_prices = close_prices - np.random.uniform(0.5, 2.0, 60)
            open_prices = low_prices + np.random.uniform(0, 1, 60) * (high_prices - low_prices)
            
            # Generate volume data
            volume = np.random.uniform(1000, 10000, 60)
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
            df.to_csv(f"{self.test_data_dir}/{symbol}/historical.csv")
            
            print(f"Generated historical data for {symbol}: {len(df)} rows")
    
    def _generate_options_data(self):
        """Generate synthetic options data"""
        # Base parameters for data generation
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        current_date = datetime.now()
        
        for symbol in symbols:
            # Get the last price from historical data
            try:
                historical_df = pd.read_csv(f"{self.test_data_dir}/{symbol}/historical.csv", index_col=0)
                current_price = historical_df['close'].iloc[-1]
            except:
                current_price = 100.0
            
            # Generate options with different strikes and expirations
            options_data = []
            
            # Define expiration dates (30, 60, 90 days out)
            expiration_dates = [
                current_date + timedelta(days=30),
                current_date + timedelta(days=60),
                current_date + timedelta(days=90)
            ]
            
            # Define strike prices (80% to 120% of current price)
            strike_multipliers = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
            strike_prices = [round(current_price * m, 2) for m in strike_multipliers]
            
            # Generate option data for each combination
            for expiration_date in expiration_dates:
                days_to_expiration = (expiration_date - current_date).days
                
                for strike_price in strike_prices:
                    # Calculate option metrics
                    itm_pct = (current_price - strike_price) / strike_price  # For calls
                    
                    # Generate CALL option
                    call_option = self._generate_option_data(
                        symbol=symbol,
                        option_type='CALL',
                        strike_price=strike_price,
                        expiration_date=expiration_date,
                        days_to_expiration=days_to_expiration,
                        current_price=current_price,
                        itm_pct=itm_pct
                    )
                    options_data.append(call_option)
                    
                    # Generate PUT option
                    put_option = self._generate_option_data(
                        symbol=symbol,
                        option_type='PUT',
                        strike_price=strike_price,
                        expiration_date=expiration_date,
                        days_to_expiration=days_to_expiration,
                        current_price=current_price,
                        itm_pct=-itm_pct  # Reverse for puts
                    )
                    options_data.append(put_option)
            
            # Create DataFrame
            options_df = pd.DataFrame(options_data)
            
            # Save to CSV
            options_df.to_csv(f"{self.test_data_dir}/{symbol}/options.csv", index=False)
            
            print(f"Generated options data for {symbol}: {len(options_df)} rows")
    
    def _generate_option_data(self, symbol, option_type, strike_price, expiration_date, days_to_expiration, current_price, itm_pct):
        """Generate data for a single option"""
        # Calculate option symbol
        option_symbol = f"{symbol}_{expiration_date.strftime('%y%m%d')}_{option_type[0]}_{int(strike_price)}"
        
        # Calculate implied volatility (higher for longer-dated options)
        base_iv = 0.3  # 30% base IV
        iv_adjustment = 0.05 * (days_to_expiration / 30)  # 5% more IV per month
        iv = base_iv + iv_adjustment
        
        # Calculate IV rank (random between 0-100)
        iv_rank = np.random.randint(0, 100)
        
        # Calculate option price based on a simple model
        time_value = current_price * iv * np.sqrt(days_to_expiration / 365)
        intrinsic_value = max(0, itm_pct * strike_price) if option_type == 'CALL' else max(0, -itm_pct * strike_price)
        theoretical_price = intrinsic_value + time_value
        
        # Add some bid-ask spread
        spread_pct = 0.05 + (0.1 * (1 - abs(itm_pct)))  # Wider spreads for OTM options
        bid = theoretical_price * (1 - spread_pct/2)
        ask = theoretical_price * (1 + spread_pct/2)
        
        # Calculate Greeks
        delta = 0.5 + (0.5 * itm_pct) if option_type == 'CALL' else 0.5 - (0.5 * itm_pct)
        delta = max(0.01, min(0.99, delta))  # Bound between 0.01 and 0.99
        
        gamma = (0.1 / strike_price) * (1 - abs(itm_pct))
        
        theta = -theoretical_price * 0.1 / days_to_expiration
        
        vega = theoretical_price * days_to_expiration / 365
        
        rho = theoretical_price * 0.05 * days_to_expiration / 365
        
        # Calculate probability of profit
        pop = 1 - delta if option_type == 'CALL' else delta
        
        # Calculate risk-reward ratio
        risk = theoretical_price
        reward = strike_price - theoretical_price if option_type == 'CALL' else theoretical_price
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Calculate liquidity score
        volume = np.random.randint(10, 1000) * (1 - 0.5 * abs(itm_pct))  # Higher volume for ATM options
        open_interest = volume * np.random.randint(5, 20)
        liquidity_score = min(1.0, (volume / 500) * 0.5 + (open_interest / 5000) * 0.5)
        
        # Calculate expected move
        expected_move = current_price * iv * np.sqrt(days_to_expiration / 365)
        
        return {
            'symbol': option_symbol,
            'underlyingSymbol': symbol,
            'optionType': option_type,
            'strikePrice': strike_price,
            'expirationDate': expiration_date.strftime('%Y-%m-%d'),
            'daysToExpiration': days_to_expiration,
            'bid': round(bid, 2),
            'ask': round(ask, 2),
            'last': round((bid + ask) / 2, 2),
            'volume': int(volume),
            'openInterest': int(open_interest),
            'impliedVolatility': iv,
            'ivRank': iv_rank,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'underlyingPrice': current_price,
            'probabilityOfProfit': pop,
            'riskRewardRatio': risk_reward_ratio,
            'potentialReturn': risk_reward_ratio * 100,
            'liquidityScore': liquidity_score,
            'expectedMove': expected_move
        }
    
    def get_historical_data(self, symbol, lookback_days=30, period_type=None, period=None, frequency_type=None, frequency=None):
        """Mock implementation of get_historical_data that returns test data"""
        try:
            df = pd.read_csv(f"{self.test_data_dir}/{symbol}/historical.csv", index_col=0, parse_dates=True)
            return df.tail(lookback_days)
        except FileNotFoundError:
            print(f"Test data not found for {symbol}")
            return pd.DataFrame()
    
    def get_options_chain(self, symbol):
        """Mock implementation of get_options_chain that returns test data"""
        try:
            df = pd.read_csv(f"{self.test_data_dir}/{symbol}/options.csv")
            return df
        except FileNotFoundError:
            print(f"Options data not found for {symbol}")
            return pd.DataFrame()

def test_dynamic_weighting_system():
    """Test dynamic weighting system based on market conditions"""
    print("\n=== Testing Dynamic Weighting System ===")
    
    # Initialize mock data collector and recommendation engine
    data_collector = MockDataCollector()
    engine = RecommendationEngine(data_collector)
    
    # Test with different market regimes
    test_regimes = [
        {'volatility': 'normal', 'trend': 'moderate', 'event_impact': 'low'},
        {'volatility': 'high', 'trend': 'strong', 'event_impact': 'medium'},
        {'volatility': 'low', 'trend': 'weak', 'event_impact': 'high'}
    ]
    
    # Test with different strategy types
    strategy_combinations = [
        None,  # All strategies
        ['directional'],
        ['income'],
        ['volatility'],
        ['directional', 'income'],
        ['directional', 'volatility'],
        ['income', 'volatility'],
        ['directional', 'income', 'volatility']
    ]
    
    results = []
    
    for regime in test_regimes:
        # Set market regime
        engine.current_volatility_regime = regime['volatility']
        engine.trend_strength = regime['trend']
        engine.market_event_impact = regime['event_impact']
        
        print(f"\nMarket Regime: {regime}")
        
        for strategies in strategy_combinations:
            # Update dynamic weights
            engine._update_dynamic_weights(strategies)
            
            # Print results
            print(f"\nStrategies: {strategies if strategies else 'All'}")
            print("Dynamic Weights:")
            for factor, weight in engine.current_weights.items():
                print(f"  {factor}: {weight:.2f}")
            
            # Store results for plotting
            for factor, weight in engine.current_weights.items():
                results.append({
                    'regime': f"{regime['volatility']}/{regime['trend']}",
                    'strategies': str(strategies),
                    'factor': factor,
                    'weight': weight
                })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Plot results for a few key combinations
    plt.figure(figsize=(12, 8))
    
    # Filter for specific combinations to make the plot readable
    plot_data = results_df[
        (results_df['strategies'].isin(['None', "['directional']", "['income']", "['volatility']"])) &
        (results_df['regime'].isin(['normal/moderate', 'high/strong', 'low/weak']))
    ]
    
    # Pivot data for plotting
    pivot_data = plot_data.pivot_table(
        index=['regime', 'strategies'],
        columns='factor',
        values='weight'
    ).reset_index()
    
    # Plot as grouped bar chart
    bar_width = 0.15
    x = np.arange(len(pivot_data))
    
    factors = [col for col in pivot_data.columns if col not in ['regime', 'strategies']]
    
    for i, factor in enumerate(factors):
        plt.bar(x + i*bar_width, pivot_data[factor], width=bar_width, label=factor)
    
    plt.xlabel('Market Regime / Strategy')
    plt.ylabel('Weight')
    plt.title('Dynamic Weights by Market Regime and Strategy')
    plt.xticks(x + bar_width * (len(factors) - 1) / 2, [f"{r} - {s}" for r, s in zip(pivot_data['regime'], pivot_data['strategies'])], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tests/output/dynamic_weights.png')
    
    return True

def test_confidence_score_calibration():
    """Test confidence score calibration based on market conditions"""
    print("\n=== Testing Confidence Score Calibration ===")
    
    # Initialize mock data collector and recommendation engine
    data_collector = MockDataCollector()
    engine = RecommendationEngine(data_collector)
    
    # Test with different market regimes
    test_regimes = [
        {'volatility': 'normal', 'trend': 'moderate', 'event_impact': 'low'},
        {'volatility': 'high', 'trend': 'strong', 'event_impact': 'medium'},
        {'volatility': 'low', 'trend': 'weak', 'event_impact': 'high'}
    ]
    
    # Test with different strategies
    strategies = ['Directional Bullish', 'Income (High Delta)', 'Long Volatility']
    
    # Test with different confluence strengths
    confluence_strengths = ['weak', 'moderate', 'strong']
    
    # Test with different raw scores
    raw_scores = [30, 50, 70, 90]
    
    results = []
    
    for regime in test_regimes:
        # Set market regime
        engine.current_volatility_regime = regime['volatility']
        engine.trend_strength = regime['trend']
        engine.market_event_impact = regime['event_impact']
        
        print(f"\nMarket Regime: {regime}")
        
        for strategy in strategies:
            for confluence in confluence_strengths:
                for raw_score in raw_scores:
                    # Calibrate confidence score
                    calibrated_score = engine._calibrate_confidence_score(raw_score, strategy, confluence)
                    
                    # Print results
                    print(f"  Strategy: {strategy}, Confluence: {confluence}, Raw Score: {raw_score}, Calibrated: {calibrated_score:.2f}")
                    
                    # Store results for plotting
                    results.append({
                        'regime': f"{regime['volatility']}/{regime['trend']}/{regime['event_impact']}",
                        'strategy': strategy,
                        'confluence': confluence,
                        'raw_score': raw_score,
                        'calibrated_score': calibrated_score
                    })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot calibration curves for different regimes (using Directional Bullish strategy and weak confluence)
    filtered_data = results_df[
        (results_df['strategy'] == 'Directional Bullish') &
        (results_df['confluence'] == 'weak')
    ]
    
    for regime in filtered_data['regime'].unique():
        regime_data = filtered_data[filtered_data['regime'] == regime]
        plt.plot(regime_data['raw_score'], regime_data['calibrated_score'], marker='o', label=regime)
    
    plt.xlabel('Raw Score')
    plt.ylabel('Calibrated Confidence Score')
    plt.title('Confidence Score Calibration by Market Regime')
    plt.legend()
    plt.grid(True)
    plt.savefig('tests/output/confidence_calibration_by_regime.png')
    
    # Plot effect of confluence strength
    plt.figure(figsize=(12, 8))
    
    filtered_data = results_df[
        (results_df['strategy'] == 'Directional Bullish') &
        (results_df['regime'] == 'normal/moderate/low')
    ]
    
    for confluence in filtered_data['confluence'].unique():
        confluence_data = filtered_data[filtered_data['confluence'] == confluence]
        plt.plot(confluence_data['raw_score'], confluence_data['calibrated_score'], marker='o', label=f"Confluence: {confluence}")
    
    plt.xlabel('Raw Score')
    plt.ylabel('Calibrated Confidence Score')
    plt.title('Effect of Signal Confluence on Confidence Calibration')
    plt.legend()
    plt.grid(True)
    plt.savefig('tests/output/confidence_calibration_by_confluence.png')
    
    return True

def test_strategy_specific_scoring():
    """Test strategy-specific scoring models"""
    print("\n=== Testing Strategy-Specific Scoring Models ===")
    
    # Initialize mock data collector and recommendation engine
    data_collector = MockDataCollector()
    engine = RecommendationEngine(data_collector)
    
    # Test with a sample symbol
    symbol = 'AAPL'
    
    # Test with different strategy types
    strategy_combinations = [
        None,  # All strategies
        ['directional'],
        ['income'],
        ['volatility']
    ]
    
    results = []
    
    for strategies in strategy_combinations:
        print(f"\nTesting with strategies: {strategies if strategies else 'All'}")
        
        # Generate recommendations
        recommendations = engine.generate_recommendations(symbol, strategy_types=strategies)
        
        if recommendations.empty:
            print("  No recommendations generated")
            continue
        
        # Print summary
        print(f"  Generated {len(recommendations)} recommendations")
        print(f"  Top recommendation: {recommendations.iloc[0]['symbol']} ({recommendations.iloc[0]['strategy']})")
        print(f"  Confidence: {recommendations.iloc[0]['confidence']:.2f}")
        
        # Count recommendations by strategy
        strategy_counts = recommendations['strategy'].value_counts()
        print("  Strategy distribution:")
        for strategy, count in strategy_counts.items():
            print(f"    {strategy}: {count}")
        
        # Store results for plotting
        for _, row in recommendations.iterrows():
            results.append({
                'strategies_filter': str(strategies),
                'recommended_strategy': row['strategy'],
                'confidence': row['confidence'],
                'option_type': row['optionType'],
                'days_to_expiration': row['daysToExpiration']
            })
    
    # Create DataFrame from results
    if results:
        results_df = pd.DataFrame(results)
        
        # Plot strategy distribution
        plt.figure(figsize=(12, 8))
        
        # Count strategies for each filter
        strategy_counts = pd.crosstab(results_df['strategies_filter'], results_df['recommended_strategy'])
        
        # Plot as stacked bar chart
        strategy_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.xlabel('Strategy Filter')
        plt.ylabel('Count')
        plt.title('Distribution of Recommended Strategies by Filter')
        plt.legend(title='Recommended Strategy')
        plt.tight_layout()
        plt.savefig('tests/output/strategy_distribution.png')
        
        # Plot confidence by strategy
        plt.figure(figsize=(12, 8))
        
        # Group by strategy and calculate mean confidence
        confidence_by_strategy = results_df.groupby(['strategies_filter', 'recommended_strategy'])['confidence'].mean().reset_index()
        
        # Pivot for plotting
        confidence_pivot = confidence_by_strategy.pivot(index='strategies_filter', columns='recommended_strategy', values='confidence')
        
        # Plot as grouped bar chart
        confidence_pivot.plot(kind='bar', figsize=(12, 8))
        plt.xlabel('Strategy Filter')
        plt.ylabel('Mean Confidence Score')
        plt.title('Mean Confidence Score by Strategy')
        plt.legend(title='Recommended Strategy')
        plt.tight_layout()
        plt.savefig('tests/output/confidence_by_strategy.png')
    
    return True

def test_full_recommendation_engine():
    """Test the full recommendation engine with all Phase 3 improvements"""
    print("\n=== Testing Full Recommendation Engine with Phase 3 Improvements ===")
    
    # Initialize mock data collector and recommendation engine
    data_collector = MockDataCollector()
    engine = RecommendationEngine(data_collector)
    
    # Test with a sample symbol
    symbol = 'AAPL'
    
    # Generate recommendations with default settings
    recommendations = engine.generate_recommendations(symbol)
    
    if recommendations.empty:
        print("No recommendations generated")
        return False
    
    # Print summary
    print(f"Generated {len(recommendations)} recommendations")
    print(f"Top recommendation: {recommendations.iloc[0]['symbol']} ({recommendations.iloc[0]['strategy']})")
    print(f"Confidence: {recommendations.iloc[0]['confidence']:.2f}")
    
    # Print details of top recommendation
    print("\nTop recommendation details:")
    top_rec = recommendations.iloc[0]
    print(f"  Symbol: {top_rec['symbol']}")
    print(f"  Option Type: {top_rec['optionType']}")
    print(f"  Strike Price: {top_rec['strikePrice']}")
    print(f"  Expiration Date: {top_rec['expirationDate']}")
    print(f"  Days to Expiration: {top_rec['daysToExpiration']}")
    print(f"  Entry Price: {top_rec['entryPrice']:.2f}")
    print(f"  Underlying Price: {top_rec['underlyingPrice']:.2f}")
    print(f"  Delta: {top_rec['delta']:.2f}")
    print(f"  Theta: {top_rec['theta']:.2f}")
    print(f"  Probability of Profit: {top_rec['probabilityOfProfit']:.2f}")
    print(f"  Risk-Reward Ratio: {top_rec['riskRewardRatio']:.2f}")
    print(f"  Raw Score: {top_rec['rawScore']:.2f}")
    print(f"  Confidence: {top_rec['confidence']:.2f}")
    print(f"  Market Direction: {top_rec['marketDirection']}")
    print(f"  Strategy: {top_rec['strategy']}")
    print(f"  Market Regime: {top_rec['marketRegime']}")
    print(f"  Confluence Strength: {top_rec['confluenceStrength']}")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(recommendations['confidence'], bins=20, alpha=0.7)
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig('tests/output/confidence_distribution.png')
    
    # Plot recommendations by strategy
    plt.figure(figsize=(10, 6))
    strategy_counts = recommendations['strategy'].value_counts()
    strategy_counts.plot(kind='bar')
    plt.xlabel('Strategy')
    plt.ylabel('Count')
    plt.title('Recommendations by Strategy')
    plt.tight_layout()
    plt.savefig('tests/output/recommendations_by_strategy.png')
    
    # Plot recommendations by option type and days to expiration
    plt.figure(figsize=(10, 6))
    plt.scatter(
        recommendations['daysToExpiration'],
        recommendations['confidence'],
        c=recommendations['optionType'].map({'CALL': 'green', 'PUT': 'red'}),
        alpha=0.7
    )
    plt.xlabel('Days to Expiration')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Score by Days to Expiration and Option Type')
    plt.colorbar(label='Option Type')
    plt.grid(True, alpha=0.3)
    plt.savefig('tests/output/confidence_by_expiration.png')
    
    return True

def run_all_tests():
    """Run all Phase 3 tests"""
    print("=== Running All Phase 3 Tests ===")
    
    # Create output directory if it doesn't exist
    os.makedirs('tests/output', exist_ok=True)
    
    tests = [
        test_dynamic_weighting_system,
        test_confidence_score_calibration,
        test_strategy_specific_scoring,
        test_full_recommendation_engine
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
            import traceback
            traceback.print_exc()
    
    if all_passed:
        print("\n=== All Phase 3 tests passed! ===")
    else:
        print("\n=== Some Phase 3 tests failed! ===")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
