"""
Test script for validating the enhanced ML and risk management features.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime as dt
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.ml.enhanced.feature_manager import FeatureEngineeringManager
from app.ml.enhanced.ensemble_models import ModelManager, StackedEnsembleModel, WeightedEnsembleModel, BoostedEnsembleModel
from app.ml.enhanced.online_learning import OnlineLearningManager, OnlineEnsembleModel, AdaptiveModelSelector
from app.ml.enhanced.risk_management import PositionSizer, StopLossTakeProfitCalculator, PortfolioRiskManager, RiskManagementIntegrator
from app.ml.enhanced.trading_system import EnhancedTradingSystem

def load_test_data(file_path=None):
    """
    Load test data for validation.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to test data file
        
    Returns:
    --------
    pandas.DataFrame
        Test data
    """
    # Use sample data if file path not provided
    if file_path is None or not os.path.exists(file_path):
        print("Creating synthetic test data...")
        # Create synthetic options data
        np.random.seed(42)
        n_samples = 1000
        
        # Create dates
        today = dt.datetime.now().date()
        dates = [today + dt.timedelta(days=i) for i in range(30, 60)]
        
        # Create synthetic data
        data = {
            'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB'], n_samples),
            'putCall': np.random.choice(['CALL', 'PUT'], n_samples),
            'strike': np.random.uniform(100, 200, n_samples),
            'bid': np.random.uniform(1, 10, n_samples),
            'ask': np.random.uniform(1.5, 11, n_samples),
            'underlyingPrice': np.random.uniform(100, 200, n_samples),
            'expirationDate': np.random.choice(dates, n_samples),
            'daysToExpiration': np.random.randint(1, 30, n_samples),
            'delta': np.random.uniform(-1, 1, n_samples),
            'gamma': np.random.uniform(0, 0.1, n_samples),
            'theta': np.random.uniform(-1, 0, n_samples),
            'vega': np.random.uniform(0, 1, n_samples),
            'rho': np.random.uniform(-0.5, 0.5, n_samples),
            'impliedVolatility': np.random.uniform(0.1, 0.5, n_samples),
            'volume': np.random.randint(1, 1000, n_samples),
            'openInterest': np.random.randint(10, 5000, n_samples),
        }
        
        # Create target variable (expected return)
        # Simulate relationship with features
        expected_return = (
            0.5 * data['delta'] + 
            2.0 * data['gamma'] - 
            0.1 * data['theta'] + 
            0.3 * data['vega'] + 
            0.1 * np.random.randn(n_samples)  # Add noise
        )
        
        data['expected_return'] = expected_return
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert date columns to datetime
        df['expirationDate'] = pd.to_datetime(df['expirationDate'])
        
        return df
    else:
        # Load data from file
        print(f"Loading data from {file_path}...")
        return pd.read_csv(file_path)

def test_feature_engineering(test_data):
    """
    Test feature engineering pipeline.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data
        
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Testing Feature Engineering Pipeline ===")
    
    # Initialize feature engineering manager
    feature_manager = FeatureEngineeringManager()
    
    # Process data
    X = test_data.drop('expected_return', axis=1)
    y = test_data['expected_return']
    
    print(f"Input data shape: {X.shape}")
    
    # Process data with feature engineering
    start_time = dt.datetime.now()
    X_processed = feature_manager.process_options_data(X, y)
    end_time = dt.datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Get feature importance
    importance = feature_manager.get_feature_importance(X, y)
    top_features = feature_manager.get_top_features(X, y, n=10)
    
    print("\nTop 10 features:")
    for feature in top_features:
        print(f"- {feature}")
    
    # Create feature report
    report = feature_manager.create_feature_report(X, y)
    
    # Compile results
    results = {
        'input_shape': X.shape,
        'output_shape': X_processed.shape,
        'processing_time': processing_time,
        'top_features': top_features,
        'feature_report': report
    }
    
    print("Feature engineering test completed successfully.")
    return results

def test_ensemble_models(test_data):
    """
    Test ensemble models.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data
        
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Testing Ensemble Models ===")
    
    # Initialize feature engineering manager
    feature_manager = FeatureEngineeringManager()
    
    # Process data
    X = test_data.drop('expected_return', axis=1)
    y = test_data['expected_return']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Process data with feature engineering
    X_train_processed = feature_manager.process_options_data(X_train, y_train)
    X_test_processed = feature_manager.process_options_data(X_test, fit=False)
    
    print(f"Training data shape: {X_train_processed.shape}")
    print(f"Test data shape: {X_test_processed.shape}")
    
    # Initialize model manager
    model_manager = ModelManager(model_dir='test_models')
    os.makedirs('test_models', exist_ok=True)
    
    # Train models
    model_results = {}
    
    # Test stacked ensemble
    print("\nTraining Stacked Ensemble Model...")
    stacked_name = model_manager.train_model('stacked', X_train_processed, y_train)
    stacked_metrics = model_manager.evaluate_model(stacked_name, X_test_processed, y_test)
    model_results['stacked'] = {
        'model_name': stacked_name,
        'metrics': stacked_metrics
    }
    print(f"Stacked Ensemble RMSE: {stacked_metrics['rmse']:.4f}")
    
    # Test weighted ensemble
    print("\nTraining Weighted Ensemble Model...")
    weighted_name = model_manager.train_model('weighted', X_train_processed, y_train)
    weighted_metrics = model_manager.evaluate_model(weighted_name, X_test_processed, y_test)
    model_results['weighted'] = {
        'model_name': weighted_name,
        'metrics': weighted_metrics
    }
    print(f"Weighted Ensemble RMSE: {weighted_metrics['rmse']:.4f}")
    
    # Test boosted ensemble
    print("\nTraining Boosted Ensemble Model...")
    boosted_name = model_manager.train_model('boosted', X_train_processed, y_train)
    boosted_metrics = model_manager.evaluate_model(boosted_name, X_test_processed, y_test)
    model_results['boosted'] = {
        'model_name': boosted_name,
        'metrics': boosted_metrics
    }
    print(f"Boosted Ensemble RMSE: {boosted_metrics['rmse']:.4f}")
    
    # Get best model
    best_name, best_model = model_manager.get_best_model(metric='rmse')
    print(f"\nBest model: {best_name}")
    
    # Save models
    for model_name in [stacked_name, weighted_name, boosted_name]:
        model_path = model_manager.save_model(model_name)
        print(f"Saved model {model_name} to {model_path}")
    
    # Compile results
    results = {
        'model_results': model_results,
        'best_model': best_name,
        'best_metrics': model_manager.metrics[best_name]
    }
    
    print("Ensemble models test completed successfully.")
    return results

def test_online_learning(test_data):
    """
    Test online learning components.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data
        
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Testing Online Learning Components ===")
    
    # Initialize feature engineering manager
    feature_manager = FeatureEngineeringManager()
    
    # Process data
    X = test_data.drop('expected_return', axis=1)
    y = test_data['expected_return']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Process data with feature engineering
    X_train_processed = feature_manager.process_options_data(X_train, y_train)
    X_test_processed = feature_manager.process_options_data(X_test, fit=False)
    
    # Initialize online learning manager
    online_manager = OnlineLearningManager(model_dir='test_online_models')
    os.makedirs('test_online_models', exist_ok=True)
    
    # Create online model
    print("\nCreating Online Ensemble Model...")
    model_name = online_manager.create_model()
    
    # Train model in batches to simulate online learning
    batch_size = 50
    n_samples = len(X_train_processed)
    
    metrics_history = []
    
    print("\nTraining Online Model in Batches...")
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        X_batch = X_train_processed.iloc[i:end_idx]
        y_batch = y_train.iloc[i:end_idx]
        
        # Update model
        metrics = online_manager.update_model(model_name, X_batch, y_batch)
        metrics_history.append(metrics)
        
        if (i // batch_size) % 5 == 0:
            print(f"Batch {i//batch_size + 1}: MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Test model
    print("\nTesting Online Model...")
    predictions = online_manager.models[model_name].predict(X_test_processed)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Final Test Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Save model
    model_path = online_manager.save_model(model_name)
    print(f"Saved online model to {model_path}")
    
    # Test adaptive model selector
    print("\nTesting Adaptive Model Selector...")
    
    # Initialize model manager for ensemble models
    model_manager = ModelManager(model_dir='test_models')
    
    # Initialize adaptive model selector
    model_selector = AdaptiveModelSelector(
        ensemble_manager=model_manager,
        online_manager=online_manager,
        performance_window=50
    )
    
    # Update performance with some test data
    for i in range(min(100, len(X_test_processed))):
        # Get predictions from different models
        online_pred = online_manager.models[model_name].predict_one(X_test_processed.iloc[i].to_dict())
        
        # Update performance
        error = abs(online_pred - y_test.iloc[i])
        model_selector.update_performance(model_name, error)
    
    # Get best model
    best_model, best_weight = model_selector.get_best_model()
    print(f"Best model according to adaptive selector: {best_model} (weight: {best_weight:.4f})")
    
    # Compile results
    results = {
        'model_name': model_name,
        'metrics_history': metrics_history,
        'final_metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'best_model': best_model,
        'best_weight': best_weight
    }
    
    print("Online learning test completed successfully.")
    return results

def test_risk_management(test_data):
    """
    Test risk management components.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data
        
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Testing Risk Management Components ===")
    
    # Sample option data
    sample_option = test_data.iloc[0].to_dict()
    
    # Test position sizer
    print("\nTesting Position Sizer...")
    position_sizer = PositionSizer(account_size=100000, max_risk_pct=2.0)
    position_size = position_sizer.calculate_position_size(sample_option, confidence_score=0.7)
    
    print(f"Recommended contracts: {position_size['recommended_contracts']}")
    print(f"Total cost: ${position_size['total_cost']:.2f}")
    print(f"Total risk: ${position_size['total_risk']:.2f}")
    print(f"Risk percentage: {position_size['risk_percentage']:.2f}%")
    
    # Test stop-loss/take-profit calculator
    print("\nTesting Stop-Loss/Take-Profit Calculator...")
    exit_calculator = StopLossTakeProfitCalculator(risk_reward_ratio=2.0)
    exit_points = exit_calculator.calculate_exit_points(sample_option)
    
    print(f"Entry price: ${exit_points['entry_price']:.2f}")
    print(f"Stop-loss: ${exit_points['final_stop_loss']:.2f}")
    print(f"Take-profit: ${exit_points['final_take_profit']:.2f}")
    print(f"Risk-reward ratio: {exit_points['risk_reward_ratio']:.2f}")
    
    # Test portfolio risk manager
    print("\nTesting Portfolio Risk Manager...")
    portfolio_manager = PortfolioRiskManager(max_portfolio_risk=15.0)
    
    # Add sample positions
    positions = []
    for i in range(5):
        option = test_data.iloc[i].to_dict()
        position_size = position_sizer.calculate_position_size(option, confidence_score=0.6 + i*0.1)
        exit_points = exit_calculator.calculate_exit_points(option)
        
        position = {
            'symbol': option.get('symbol', f"SYM{i}"),
            'option_type': option.get('putCall', 'CALL'),
            'strike': option.get('strike', 100),
            'expiration': str(option.get('expirationDate', '2023-12-31')),
            'contracts': position_size.get('recommended_contracts', 1),
            'exposure': position_size.get('total_cost', 1000),
            'risk': position_size.get('total_risk', 500),
            'sector': ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy'][i % 5],
            'strategy': ['Momentum', 'Value', 'Growth', 'Income', 'Volatility'][i % 5],
            'expected_return': 0.2 + i*0.1,
            'confidence': 0.6 + i*0.1,
            'stop_loss': exit_points.get('final_stop_loss', 0),
            'take_profit': exit_points.get('final_take_profit', 0)
        }
        
        position_id = portfolio_manager.add_position(position)
        positions.append(position)
        print(f"Added position {position_id}: {position['symbol']} {position['option_type']} ${position['strike']}")
    
    # Calculate portfolio risk
    portfolio_risk = portfolio_manager.calculate_portfolio_risk()
    
    print(f"Total positions: {portfolio_risk['total_positions']}")
    print(f"Total exposure: ${portfolio_risk['total_exposure']:.2f}")
    print(f"Total risk: ${portfolio_risk['total_risk']:.2f}")
    print(f"Diversification score: {portfolio_risk['diversification_score']:.2f}/100")
    
    # Generate risk report
    risk_report = portfolio_manager.generate_risk_report(account_size=100000)
    
    # Test risk management integrator
    print("\nTesting Risk Management Integrator...")
    risk_integrator = RiskManagementIntegrator(account_size=100000, risk_profile='moderate')
    
    # Process a recommendation
    recommendation = {
        'symbol': sample_option.get('symbol', 'AAPL'),
        'option_data': sample_option,
        'prediction': 0.8,
        'confidence': {
            'score': 0.75,
            'model': 'test_model'
        },
        'timestamp': dt.datetime.now().isoformat()
    }
    
    enhanced_rec = risk_integrator.process_recommendation(recommendation)
    
    print(f"Processed recommendation for {enhanced_rec['symbol']}")
    print(f"Recommended contracts: {enhanced_rec['risk_management']['position_sizing']['recommended_contracts']}")
    print(f"Stop-loss: ${enhanced_rec['risk_management']['exit_points']['final_stop_loss']:.2f}")
    print(f"Take-profit: ${enhanced_rec['risk_management']['exit_points']['final_take_profit']:.2f}")
    
    # Compile results
    results = {
        'position_sizing': position_size,
        'exit_points': exit_points,
        'portfolio_risk': portfolio_risk,
        'risk_report': risk_report,
        'enhanced_recommendation': enhanced_rec
    }
    
    print("Risk management test completed successfully.")
    return results

def test_trading_system(test_data):
    """
    Test the integrated trading system.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data
        
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Testing Integrated Trading System ===")
    
    # Initialize trading system
    trading_system = EnhancedTradingSystem()
    
    # Process data
    X = test_data.drop('expected_return', axis=1)
    y = test_data['expected_return']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    print("\nTraining models...")
    trained_models = trading_system.train_models(X_train, y_train)
    
    print(f"Trained {len(trained_models)} models:")
    for model_type, details in trained_models.items():
        print(f"- {model_type}: {details['model_name']}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    prediction_result = trading_system.predict(X_test)
    predictions = prediction_result['predictions']
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Prediction metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"Selected model: {prediction_result['model_details']['selected_model']}")
    
    # Update online model
    print("\nUpdating online model...")
    update_result = trading_system.update_online_model(X_test.iloc[:100], y_test.iloc[:100])
    
    print(f"Updated model: {update_result['model_name']}")
    print(f"Updated metrics - MAE: {update_result['metrics']['mae']:.4f}, RMSE: {update_result['metrics']['rmse']:.4f}")
    
    # Generate trading recommendations
    print("\nGenerating trading recommendations...")
    recommendations = trading_system.generate_trading_recommendation(X_test.iloc[:10])
    
    print(f"Generated {len(recommendations)} recommendations")
    
    if recommendations:
        sample_rec = recommendations[0]
        print(f"\nSample recommendation for {sample_rec['symbol']}:")
        print(f"Prediction: {sample_rec['prediction']:.4f}")
        print(f"Confidence: {sample_rec['confidence']['score']:.4f}")
        print(f"Recommended contracts: {sample_rec['risk_management']['position_sizing']['recommended_contracts']}")
        print(f"Stop-loss: ${sample_rec['risk_management']['exit_points']['final_stop_loss']:.2f}")
        print(f"Take-profit: ${sample_rec['risk_management']['exit_points']['final_take_profit']:.2f}")
    
    # Generate portfolio report
    print("\nGenerating portfolio report...")
    portfolio_report = trading_system.generate_portfolio_report()
    
    print(f"Portfolio has {portfolio_report.get('total_positions', 0)} positions")
    print(f"Total exposure: ${portfolio_report.get('total_exposure', 0):.2f}")
    print(f"Total risk: ${portfolio_report.get('total_risk', 0):.2f}")
    
    # Save configuration
    config_path = 'trading_system_config.json'
    saved_path = trading_system.save_configuration(config_path)
    print(f"\nSaved configuration to {saved_path}")
    
    # Compile results
    results = {
        'trained_models': trained_models,
        'prediction_metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'online_update': update_result,
        'recommendations': recommendations,
        'portfolio_report': portfolio_report,
        'config_path': saved_path
    }
    
    print("Integrated trading system test completed successfully.")
    return results

def run_all_tests():
    """
    Run all tests and generate a comprehensive report.
    
    Returns:
    --------
    dict
        Complete test results
    """
    print("=== Starting Comprehensive Testing of ML and Risk Management Enhancements ===")
    
    # Create results directory
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load test data
    test_data = load_test_data()
    
    # Run tests
    feature_results = test_feature_engineering(test_data)
    ensemble_results = test_ensemble_models(test_data)
    online_results = test_online_learning(test_data)
    risk_results = test_risk_management(test_data)
    system_results = test_trading_system(test_data)
    
    # Compile all results
    all_results = {
        'feature_engineering': feature_results,
        'ensemble_models': ensemble_results,
        'online_learning': online_results,
        'risk_management': risk_results,
        'trading_system': system_results,
        'timestamp': dt.datetime.now().isoformat()
    }
    
    # Save results
    results_path = os.path.join(results_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    
    print(f"\nAll tests completed successfully. Results saved to {results_path}")
    
    return all_results

if __name__ == "__main__":
    run_all_tests()
