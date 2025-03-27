"""
Test script for the feature engineering pipeline.
Tests data preprocessing, feature extraction, and feature selection components.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.feature_engineering import (
    DataPreprocessor, 
    FeatureExtractor, 
    FeatureSelector,
    FeatureEngineeringPipeline
)

def generate_test_data(n_samples=100):
    """
    Generate synthetic test data for feature engineering pipeline testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    tuple
        (X, y) where X is a DataFrame of features and y is a Series of target values
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=n_samples)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    # Generate price data
    base_price = 100.0
    trend = np.linspace(0, 20, n_samples)  # Upward trend
    cycles = 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))  # Cyclical component
    noise = np.random.normal(0, 1, n_samples)  # Random noise
    
    # Combine components
    close_prices = base_price + trend + cycles + noise
    
    # Generate OHLC data
    high_prices = close_prices + np.random.uniform(0.5, 2.0, n_samples)
    low_prices = close_prices - np.random.uniform(0.5, 2.0, n_samples)
    open_prices = low_prices + np.random.uniform(0, 1, n_samples) * (high_prices - low_prices)
    
    # Generate volume data
    volume = np.random.uniform(1000, 10000, n_samples)
    # Make volume correlate somewhat with price changes
    price_changes = np.diff(close_prices, prepend=close_prices[0])
    volume = volume * (1 + 0.5 * np.abs(price_changes) / np.mean(np.abs(price_changes)))
    
    # Generate technical indicators
    rsi_14 = 50 + 25 * np.sin(np.linspace(0, 6*np.pi, n_samples))  # RSI oscillating between 25 and 75
    macd = 5 * np.sin(np.linspace(0, 8*np.pi, n_samples))  # MACD oscillating between -5 and 5
    ema_9 = close_prices - 5 + 3 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    ema_21 = close_prices - 2 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    
    # Generate options data
    strike_price = close_prices[-1]
    days_to_expiration = 30
    implied_volatility = 0.3 + 0.1 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    delta = 0.5 + 0.4 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    gamma = 0.05 + 0.03 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    theta = -0.1 - 0.05 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    vega = 0.2 + 0.1 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    
    # Generate market data
    vix = 15 + 10 * np.sin(np.linspace(0, 2*np.pi, n_samples))
    market_trend = np.random.choice(['bullish', 'bearish', 'neutral'], n_samples)
    sector_trend = np.random.choice(['bullish', 'bearish', 'neutral'], n_samples)
    
    # Generate categorical data
    option_type = np.random.choice(['CALL', 'PUT'], n_samples)
    
    # Create DataFrame
    X = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume.astype(int),
        'rsi_14': rsi_14,
        'macd': macd,
        'ema_9': ema_9,
        'ema_21': ema_21,
        'strikePrice': [strike_price] * n_samples,
        'daysToExpiration': [days_to_expiration] * n_samples,
        'impliedVolatility': implied_volatility,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'vix': vix,
        'market_trend': market_trend,
        'sector_trend': sector_trend,
        'optionType': option_type,
        'underlyingPrice': close_prices
    })
    
    # Generate target variable (future price change)
    future_returns = np.roll(close_prices, -5) / close_prices - 1
    future_returns[-5:] = 0  # Set last 5 values to 0 since we don't have future data
    y = pd.Series(future_returns, name='future_return')
    
    return X, y

def test_data_preprocessor():
    """
    Test the DataPreprocessor class.
    """
    print("\n=== Testing DataPreprocessor ===")
    
    # Generate test data
    X, y = generate_test_data()
    
    # Add some missing values
    X.loc[10:15, 'close'] = np.nan
    X.loc[20:25, 'volume'] = np.nan
    X.loc[30:35, 'market_trend'] = np.nan
    
    # Add some outliers
    X.loc[40, 'close'] = X['close'].mean() + 10 * X['close'].std()
    X.loc[50, 'volume'] = X['volume'].mean() + 10 * X['volume'].std()
    
    print(f"Original data shape: {X.shape}")
    print(f"Missing values before preprocessing:")
    print(X.isna().sum().sum())
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        numerical_features=None,  # Auto-detect
        categorical_features=None,  # Auto-detect
        date_features=None,  # Auto-detect
        normalization='standard',
        handle_outliers=True
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Missing values after preprocessing:")
    print(X_processed.isna().sum().sum())
    
    # Check if outliers were handled
    if 'close' in X_processed.columns:
        print(f"Original close range: [{X['close'].min()}, {X['close'].max()}]")
        print(f"Processed close range: [{X_processed['close'].min()}, {X_processed['close'].max()}]")
    
    # Plot original vs processed data for a numerical feature
    plt.figure(figsize=(12, 6))
    
    if isinstance(X_processed, pd.DataFrame) and 'close' in X_processed.columns:
        plt.subplot(1, 2, 1)
        plt.plot(X['close'])
        plt.title('Original Close Prices')
        plt.xlabel('Sample')
        plt.ylabel('Price')
        
        plt.subplot(1, 2, 2)
        plt.plot(X_processed['close'])
        plt.title('Processed Close Prices')
        plt.xlabel('Sample')
        plt.ylabel('Standardized Price')
    else:
        plt.text(0.5, 0.5, 'Processed data is not a DataFrame with original column names', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('tests/output', exist_ok=True)
    plt.savefig('tests/output/preprocessor_test.png')
    
    return True

def test_feature_extractor():
    """
    Test the FeatureExtractor class.
    """
    print("\n=== Testing FeatureExtractor ===")
    
    # Generate test data
    X, y = generate_test_data()
    
    print(f"Original data shape: {X.shape}")
    print(f"Original columns: {X.columns.tolist()}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        extract_technical=True,
        extract_volatility=True,
        extract_options=True,
        extract_market=True
    )
    
    # Transform
    X_extracted = extractor.fit_transform(X)
    
    print(f"Extracted data shape: {X_extracted.shape}")
    print(f"New columns added: {set(X_extracted.columns) - set(X.columns)}")
    
    # Count new features by category
    technical_features = [col for col in X_extracted.columns if col not in X.columns and 
                         (col.endswith('_roc') or 'divergence' in col or 'crossover' in col)]
    
    volatility_features = [col for col in X_extracted.columns if col not in X.columns and 
                          ('vol_' in col or 'hist_vol' in col)]
    
    options_features = [col for col in X_extracted.columns if col not in X.columns and 
                       ('moneyness' in col or 'itm' in col or 'otm' in col)]
    
    market_features = [col for col in X_extracted.columns if col not in X.columns and 
                      ('market_' in col or 'sector_' in col or 'vix_' in col)]
    
    print(f"Technical features added: {len(technical_features)}")
    print(f"Volatility features added: {len(volatility_features)}")
    print(f"Options features added: {len(options_features)}")
    print(f"Market features added: {len(market_features)}")
    
    # Plot some of the extracted features
    plt.figure(figsize=(15, 10))
    
    # Plot technical features
    if technical_features:
        plt.subplot(2, 2, 1)
        for i, feature in enumerate(technical_features[:3]):  # Plot first 3 features
            plt.plot(X_extracted[feature], label=feature)
        plt.title('Technical Features')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
    
    # Plot volatility features
    if volatility_features:
        plt.subplot(2, 2, 2)
        for i, feature in enumerate(volatility_features[:3]):  # Plot first 3 features
            plt.plot(X_extracted[feature], label=feature)
        plt.title('Volatility Features')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
    
    # Plot options features
    if options_features:
        plt.subplot(2, 2, 3)
        for i, feature in enumerate(options_features[:3]):  # Plot first 3 features
            plt.plot(X_extracted[feature], label=feature)
        plt.title('Options Features')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
    
    # Plot market features
    if market_features:
        plt.subplot(2, 2, 4)
        for i, feature in enumerate(market_features[:3]):  # Plot first 3 features
            plt.plot(X_extracted[feature], label=feature)
        plt.title('Market Features')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('tests/output/feature_extractor_test.png')
    
    return True

def test_feature_selector():
    """
    Test the FeatureSelector class.
    """
    print("\n=== Testing FeatureSelector ===")
    
    # Generate test data
    X, y = generate_test_data()
    
    # Extract features first
    extractor = FeatureExtractor()
    X_extracted = extractor.fit_transform(X)
    
    print(f"Data shape before selection: {X_extracted.shape}")
    
    # Test different selection methods
    selection_methods = ['k_best', 'mutual_info', 'correlation']
    
    for method in selection_methods:
        print(f"\nTesting selection method: {method}")
        
        # Initialize selector
        selector = FeatureSelector(
            method=method,
            k=10,
            threshold=0.1,
            use_pca=False
        )
        
        # Fit and transform
        X_selected = selector.fit_transform(X_extracted, y)
        
        print(f"Selected data shape: {X_selected.shape}")
        
        if isinstance(X_selected, pd.DataFrame):
            print(f"Selected features: {X_selected.columns.tolist()}")
    
    # Test with PCA
    print("\nTesting with PCA")
    
    # Initialize selector with PCA
    selector_pca = FeatureSelector(
        method='k_best',
        k=20,
        use_pca=True,
        n_components=5
    )
    
    # Fit and transform
    X_pca = selector_pca.fit_transform(X_extracted, y)
    
    print(f"PCA data shape: {X_pca.shape}")
    
    if isinstance(X_pca, pd.DataFrame):
        print(f"PCA components: {X_pca.columns.tolist()}")
    
    # Plot explained variance ratio
    if hasattr(selector_pca, 'pca') and selector_pca.pca is not None:
        plt.figure(figsize=(10, 6))
        
        explained_variance = selector_pca.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
        plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
        
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.tight_layout()
        plt.savefig('tests/output/feature_selector_pca_test.png')
    
    return True

def test_full_pipeline():
    """
    Test the complete FeatureEngineeringPipeline.
    """
    print("\n=== Testing Full Feature Engineering Pipeline ===")
    
    # Generate test data
    X, y = generate_test_data()
    
    print(f"Original data shape: {X.shape}")
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(
        numerical_features=None,  # Auto-detect
        categorical_features=None,  # Auto-detect
        date_features=['date'],
        normalization='standard',
        handle_outliers=True,
        extract_technical=True,
        extract_volatility=True,
        extract_options=True,
        extract_market=True,
        selection_method='k_best',
        k=20,
        use_pca=True
    )
    
    # Fit and transform
    X_transformed = pipeline.fit_transform(X, y)
    
    print(f"Transformed data shape: {X_transformed.shape}")
    
    if isinstance(X_transformed, pd.DataFrame):
        print(f"Final features: {X_transformed.columns.tolist()}")
    
    # Plot correlation with target for original vs transformed features
    plt.figure(figsize=(12, 8))
    
    # Original correlations
    original_corr = X.select_dtypes(include=['number']).corrwith(y).abs().sort_values(ascending=False)
    top_original = original_corr.head(10)
    
    plt.subplot(1, 2, 1)
    top_original.plot(kind='bar')
    plt.title('Top 10 Original Features\nCorrelation with Target')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45, ha='right')
    
    # Transformed correlations
    if isinstance(X_transformed, pd.DataFrame):
        transformed_corr = X_transformed.corrwith(y).abs().sort_values(ascending=False)
        top_transformed = transformed_corr.head(10)
        
        plt.subplot(1, 2, 2)
        top_transformed.plot(kind='bar')
        plt.title('Top 10 Transformed Features\nCorrelation with Target')
        plt.ylabel('Absolute Correlation')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('tests/output/full_pipeline_test.png')
    
    return True

def run_all_tests():
    """
    Run all feature engineering tests.
    """
    print("=== Running All Feature Engineering Tests ===")
    
    # Create output directory if it doesn't exist
    os.makedirs('tests/output', exist_ok=True)
    
    tests = [
        test_data_preprocessor,
        test_feature_extractor,
        test_feature_selector,
        test_full_pipeline
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
        print("\n=== All feature engineering tests passed! ===")
    else:
        print("\n=== Some feature engineering tests failed! ===")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
