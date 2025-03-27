"""
Integration script for enhanced ML and risk management components.
Connects the enhanced features with the existing dashboard.
"""
import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import json
import logging

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.ml.enhanced.trading_system import EnhancedTradingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ml_integration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('enhanced_ml_integration')

class EnhancedMLIntegration:
    """
    Integration class for connecting enhanced ML and risk management features
    with the existing options trading dashboard.
    """
    def __init__(self, config_path=None):
        """
        Initialize the integration module.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.config_path = config_path
        self.trading_system = None
        self.initialize_system()
        logger.info("Enhanced ML Integration initialized")
    
    def initialize_system(self):
        """
        Initialize the enhanced trading system.
        """
        try:
            # Load configuration if provided
            if self.config_path and os.path.exists(self.config_path):
                logger.info(f"Loading configuration from {self.config_path}")
                self.trading_system = EnhancedTradingSystem()
                self.trading_system.load_configuration(self.config_path)
            else:
                logger.info("Using default configuration")
                self.trading_system = EnhancedTradingSystem()
            
            logger.info("Enhanced trading system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            raise
    
    def process_options_data(self, options_data, target=None):
        """
        Process options data through the enhanced feature engineering pipeline.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data to process
        target : array-like, optional
            Target values for supervised feature selection
            
        Returns:
        --------
        pandas.DataFrame
            Processed data with engineered features
        """
        try:
            logger.info(f"Processing options data with shape {options_data.shape}")
            processed_data = self.trading_system.process_data(options_data, target)
            logger.info(f"Processed data shape: {processed_data.shape}")
            return processed_data
        except Exception as e:
            logger.error(f"Error processing options data: {str(e)}")
            raise
    
    def train_prediction_models(self, options_data, target_column):
        """
        Train prediction models on options data.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data for training
        target_column : str
            Name of the target column
            
        Returns:
        --------
        dict
            Dictionary with trained model details
        """
        try:
            logger.info(f"Training prediction models on {len(options_data)} samples")
            
            # Extract features and target
            X = options_data.drop(target_column, axis=1)
            y = options_data[target_column]
            
            # Train models
            trained_models = self.trading_system.train_models(X, y)
            
            logger.info(f"Trained {len(trained_models)} models successfully")
            return trained_models
        except Exception as e:
            logger.error(f"Error training prediction models: {str(e)}")
            raise
    
    def generate_predictions(self, options_data):
        """
        Generate predictions for options data.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data to predict
            
        Returns:
        --------
        dict
            Dictionary with predictions and details
        """
        try:
            logger.info(f"Generating predictions for {len(options_data)} options")
            prediction_result = self.trading_system.predict(options_data)
            logger.info(f"Generated predictions using model: {prediction_result['model_details']['selected_model']}")
            return prediction_result
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def update_models_with_feedback(self, options_data, actual_values):
        """
        Update models with feedback data.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data
        actual_values : array-like
            Actual target values
            
        Returns:
        --------
        dict
            Updated metrics
        """
        try:
            logger.info(f"Updating models with feedback data ({len(options_data)} samples)")
            update_result = self.trading_system.update_online_model(options_data, actual_values)
            logger.info(f"Updated model: {update_result['model_name']}")
            return update_result
        except Exception as e:
            logger.error(f"Error updating models with feedback: {str(e)}")
            raise
    
    def generate_trading_recommendations(self, options_data, market_data=None):
        """
        Generate trading recommendations with risk management.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data to analyze
        market_data : dict, optional
            Additional market data for risk calculations
            
        Returns:
        --------
        list
            List of trading recommendations with risk management details
        """
        try:
            logger.info(f"Generating trading recommendations for {len(options_data)} options")
            recommendations = self.trading_system.generate_trading_recommendation(options_data, market_data)
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {str(e)}")
            raise
    
    def get_portfolio_risk_report(self):
        """
        Generate a comprehensive portfolio risk report.
        
        Returns:
        --------
        dict
            Portfolio risk report
        """
        try:
            logger.info("Generating portfolio risk report")
            report = self.trading_system.generate_portfolio_report()
            logger.info(f"Portfolio has {report.get('total_positions', 0)} positions")
            return report
        except Exception as e:
            logger.error(f"Error generating portfolio risk report: {str(e)}")
            raise
    
    def update_account_settings(self, account_size=None, risk_profile=None):
        """
        Update account settings for risk management.
        
        Parameters:
        -----------
        account_size : float, optional
            New account size
        risk_profile : str, optional
            New risk profile ('conservative', 'moderate', or 'aggressive')
        """
        try:
            if account_size is not None:
                logger.info(f"Updating account size to ${account_size}")
                self.trading_system.update_account_size(account_size)
            
            if risk_profile is not None:
                logger.info(f"Updating risk profile to {risk_profile}")
                self.trading_system.update_risk_profile(risk_profile)
            
            # Save updated configuration
            if self.config_path:
                self.trading_system.save_configuration(self.config_path)
                logger.info(f"Saved updated configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error updating account settings: {str(e)}")
            raise
    
    def save_current_configuration(self, config_path=None):
        """
        Save current configuration to file.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to save configuration (uses self.config_path if not provided)
            
        Returns:
        --------
        str
            Path to saved configuration
        """
        try:
            save_path = config_path or self.config_path or 'enhanced_trading_config.json'
            logger.info(f"Saving configuration to {save_path}")
            saved_path = self.trading_system.save_configuration(save_path)
            self.config_path = saved_path
            return saved_path
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # This is just an example of how to use the integration module
    try:
        # Initialize integration
        integration = EnhancedMLIntegration()
        
        # Load sample data (in a real scenario, this would come from the dashboard)
        # Create synthetic options data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB'], n_samples),
            'putCall': np.random.choice(['CALL', 'PUT'], n_samples),
            'strike': np.random.uniform(100, 200, n_samples),
            'bid': np.random.uniform(1, 10, n_samples),
            'ask': np.random.uniform(1.5, 11, n_samples),
            'underlyingPrice': np.random.uniform(100, 200, n_samples),
            'daysToExpiration': np.random.randint(1, 30, n_samples),
            'delta': np.random.uniform(-1, 1, n_samples),
            'gamma': np.random.uniform(0, 0.1, n_samples),
            'theta': np.random.uniform(-1, 0, n_samples),
            'vega': np.random.uniform(0, 1, n_samples),
            'rho': np.random.uniform(-0.5, 0.5, n_samples),
            'impliedVolatility': np.random.uniform(0.1, 0.5, n_samples),
            'volume': np.random.randint(1, 1000, n_samples),
            'openInterest': np.random.randint(10, 5000, n_samples),
            'expected_return': np.random.uniform(-0.2, 0.5, n_samples)
        }
        
        sample_data = pd.DataFrame(data)
        
        # Process data
        processed_data = integration.process_options_data(sample_data)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Generate recommendations
        recommendations = integration.generate_trading_recommendations(sample_data)
        print(f"Generated {len(recommendations)} recommendations")
        
        # Save configuration
        config_path = integration.save_current_configuration()
        print(f"Saved configuration to {config_path}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
