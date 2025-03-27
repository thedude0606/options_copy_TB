"""
Integration module for ML and risk management components.
Connects the enhanced ML and risk management features with the existing system.
"""
import pandas as pd
import numpy as np
import os
import datetime as dt
import json

from app.ml.enhanced.feature_manager import FeatureEngineeringManager
from app.ml.enhanced.ensemble_models import ModelManager, StackedEnsembleModel, WeightedEnsembleModel, BoostedEnsembleModel
from app.ml.enhanced.online_learning import OnlineLearningManager, AdaptiveModelSelector
from app.ml.enhanced.risk_management import RiskManagementIntegrator

class EnhancedTradingSystem:
    """
    Main integration class for enhanced trading system.
    Combines ML and risk management components.
    """
    def __init__(self, config=None):
        """
        Initialize the enhanced trading system.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with system settings
        """
        self.config = config or self._default_config()
        self._initialize_components()
    
    def _default_config(self):
        """
        Create default configuration for the system.
        
        Returns:
        --------
        dict
            Default configuration dictionary
        """
        return {
            'account_size': 100000,
            'risk_profile': 'moderate',
            'feature_engineering': {
                'numerical_features': [
                    'strike', 'bid', 'ask', 'underlyingPrice', 'daysToExpiration',
                    'delta', 'gamma', 'theta', 'vega', 'rho', 'impliedVolatility',
                    'volume', 'openInterest'
                ],
                'categorical_features': ['putCall'],
                'date_features': ['expirationDate'],
                'normalization': 'standard',
                'handle_outliers': True
            },
            'model_management': {
                'model_dir': 'models',
                'use_ensemble': True,
                'use_online_learning': True,
                'performance_window': 100
            },
            'risk_management': {
                'max_risk_pct': 2.0,
                'max_position_pct': 5.0,
                'risk_reward_ratio': 2.0,
                'max_portfolio_risk': 15.0
            }
        }
    
    def _initialize_components(self):
        """
        Initialize system components based on configuration.
        """
        # Initialize feature engineering manager
        self.feature_manager = FeatureEngineeringManager(
            config=self.config.get('feature_engineering')
        )
        
        # Initialize model managers
        model_config = self.config.get('model_management', {})
        self.ensemble_manager = ModelManager(
            model_dir=model_config.get('model_dir', 'models')
        )
        
        self.online_manager = OnlineLearningManager(
            model_dir=model_config.get('model_dir', 'online_models')
        )
        
        # Initialize adaptive model selector
        self.model_selector = AdaptiveModelSelector(
            ensemble_manager=self.ensemble_manager,
            online_manager=self.online_manager,
            performance_window=model_config.get('performance_window', 100)
        )
        
        # Initialize risk management integrator
        risk_config = self.config.get('risk_management', {})
        self.risk_manager = RiskManagementIntegrator(
            account_size=self.config.get('account_size', 100000),
            risk_profile=self.config.get('risk_profile', 'moderate')
        )
    
    def process_data(self, options_data, target=None, fit=True):
        """
        Process options data through the feature engineering pipeline.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data to process
        target : array-like, optional
            Target values for supervised feature selection
        fit : bool
            Whether to fit the pipeline to the data
            
        Returns:
        --------
        pandas.DataFrame
            Processed data with engineered features
        """
        return self.feature_manager.process_options_data(options_data, target, fit)
    
    def train_models(self, X, y, model_types=None):
        """
        Train prediction models on the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        y : array-like
            Target values
        model_types : list, optional
            Types of models to train
            
        Returns:
        --------
        dict
            Dictionary with trained model details
        """
        # Process data through feature engineering pipeline
        X_processed = self.process_data(X, y)
        
        # Default model types if not specified
        if model_types is None:
            model_types = ['stacked', 'weighted', 'boosted']
        
        # Train models
        trained_models = {}
        
        for model_type in model_types:
            # Train ensemble model
            model_name = self.ensemble_manager.train_model(
                model_type=model_type,
                X=X_processed,
                y=y
            )
            
            # Save model
            model_path = self.ensemble_manager.save_model(model_name)
            
            # Store details
            trained_models[model_type] = {
                'model_name': model_name,
                'model_path': model_path,
                'metrics': self.ensemble_manager.metrics.get(model_name, {})
            }
        
        # Create online learning model
        online_model_name = self.online_manager.create_model()
        
        # Train online model
        self.online_manager.update_model(online_model_name, X_processed, y)
        
        # Save online model
        online_model_path = self.online_manager.save_model(online_model_name)
        
        # Store details
        trained_models['online'] = {
            'model_name': online_model_name,
            'model_path': online_model_path,
            'metrics': self.online_manager.models[online_model_name].get_metrics()
        }
        
        return trained_models
    
    def predict(self, options_data, model_name=None):
        """
        Generate predictions for options data.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            Options data to predict
        model_name : str, optional
            Specific model to use for prediction
            
        Returns:
        --------
        dict
            Dictionary with predictions and details
        """
        # Process data through feature engineering pipeline
        X_processed = self.process_data(options_data, fit=False)
        
        # Use adaptive model selector if no specific model is specified
        if model_name is None:
            predictions, details = self.model_selector.predict(X_processed)
            selected_model = details['selected_model']
        else:
            # Use specified model
            if model_name in self.ensemble_manager.models:
                predictions = self.ensemble_manager.models[model_name].predict(X_processed)
                selected_model = model_name
            elif model_name in self.online_manager.models:
                predictions = self.online_manager.models[model_name].predict(X_processed)
                selected_model = model_name
            else:
                raise ValueError(f"Model not found: {model_name}")
            
            details = {'selected_model': selected_model}
        
        # Compile result
        result = {
            'predictions': predictions,
            'model_details': details
        }
        
        return result
    
    def update_online_model(self, options_data, actual_values, model_name=None):
        """
        Update online learning model with new data.
        
        Parameters:
        -----------
        options_data : pandas.DataFrame
            New options data
        actual_values : array-like
            Actual target values
        model_name : str, optional
            Specific model to update
            
        Returns:
        --------
        dict
            Updated metrics
        """
        # Process data through feature engineering pipeline
        X_processed = self.process_data(options_data, fit=False)
        
        # Use first available model if none specified
        if model_name is None:
            model_names = list(self.online_manager.models.keys())
            if not model_names:
                # Create new model if none exists
                model_name = self.online_manager.create_model()
            else:
                model_name = model_names[0]
        
        # Update model
        metrics = self.online_manager.update_model(model_name, X_processed, actual_values)
        
        # Save model
        self.online_manager.save_model(model_name)
        
        # Update model selector performance
        predictions = self.online_manager.models[model_name].predict(X_processed)
        errors = np.abs(predictions - actual_values)
        mean_error = np.mean(errors)
        
        self.model_selector.update_performance(model_name, mean_error)
        
        return {
            'model_name': model_name,
            'metrics': metrics,
            'mean_error': mean_error
        }
    
    def generate_trading_recommendation(self, options_data, market_data=None):
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
        # Process data through feature engineering pipeline
        X_processed = self.process_data(options_data, fit=False)
        
        # Generate predictions
        prediction_result = self.predict(options_data)
        predictions = prediction_result['predictions']
        model_details = prediction_result['model_details']
        
        # Create recommendations
        recommendations = []
        
        for i, (idx, row) in enumerate(options_data.iterrows()):
            # Skip if prediction is not positive
            if predictions[i] <= 0:
                continue
            
            # Create base recommendation
            recommendation = {
                'symbol': row.get('symbol', ''),
                'option_data': row.to_dict(),
                'prediction': predictions[i],
                'confidence': {
                    'score': min(1.0, max(0.0, predictions[i] / 10)),  # Scale to 0-1
                    'model': model_details.get('selected_model', 'unknown')
                },
                'timestamp': dt.datetime.now().isoformat()
            }
            
            # Add risk management
            enhanced_rec = self.risk_manager.process_recommendation(
                recommendation,
                market_data=market_data
            )
            
            recommendations.append(enhanced_rec)
        
        return recommendations
    
    def update_account_size(self, account_size):
        """
        Update account size for risk management.
        
        Parameters:
        -----------
        account_size : float
            New account size
        """
        self.config['account_size'] = account_size
        self.risk_manager.update_account_size(account_size)
    
    def update_risk_profile(self, risk_profile):
        """
        Update risk profile for risk management.
        
        Parameters:
        -----------
        risk_profile : str
            New risk profile ('conservative', 'moderate', or 'aggressive')
        """
        self.config['risk_profile'] = risk_profile
        self.risk_manager.update_risk_profile(risk_profile)
    
    def generate_portfolio_report(self):
        """
        Generate a comprehensive portfolio risk report.
        
        Returns:
        --------
        dict
            Portfolio risk report
        """
        return self.risk_manager.generate_portfolio_report()
    
    def save_configuration(self, config_path):
        """
        Save system configuration to file.
        
        Parameters:
        -----------
        config_path : str
            Path to save configuration
            
        Returns:
        --------
        str
            Path to saved configuration
        """
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        return config_path
    
    def load_configuration(self, config_path):
        """
        Load system configuration from file.
        
        Parameters:
        -----------
        config_path : str
            Path to load configuration from
            
        Returns:
        --------
        dict
            Loaded configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Re-initialize components with new configuration
        self._initialize_components()
        
        return self.config
