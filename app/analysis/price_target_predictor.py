"""
Price Target Prediction Module

This module provides specialized functionality for predicting optimal price targets
for options positions, determining at what premium to sell options contracts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Import utils
from app.utils.enhanced_logging import EnhancedLogger

class PriceTargetPredictor:
    """
    Specialized predictor for optimal price targets of options positions.
    
    This class focuses specifically on determining at what premium to sell options,
    using both rule-based approaches and machine learning to generate price targets
    for partial or complete exits.
    """
    
    def __init__(self, data_collector=None, config_path=None):
        """
        Initialize the price target predictor.
        
        Args:
            data_collector: Data collector for market data
            config_path: Path to configuration file
        """
        self.logger = EnhancedLogger('price_target_predictor')
        self.logger.info("Initializing Price Target Predictor")
        
        # Store data collector
        self.data_collector = data_collector
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        
        # Create model directory
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'price_targets')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load pre-trained models if available
        self._load_models()
        
        self.logger.info("Price Target Predictor initialized successfully")
    
    def _load_config(self, config_path=None):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            'profit_taking_levels': [0.25, 0.5, 0.75, 1.0],
            'position_sizing': [0.25, 0.25, 0.25, 0.25],
            'adjust_for_volatility': True,
            'volatility_adjustment_factor': 2.0,
            'min_profit_target': 0.1,  # 10% minimum profit target
            'max_profit_target': 3.0,  # 300% maximum profit target
            'risk_reward_ratio': 2.0,  # Target 2:1 reward-to-risk ratio
            'model_features': [
                'days_to_expiration', 'implied_volatility', 'delta', 'gamma', 
                'theta', 'vega', 'rho', 'option_type', 'moneyness', 'volume_oi_ratio'
            ],
            'market_features': [
                'trend_strength', 'volatility_regime', 'rsi', 'macd_histogram'
            ],
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                },
                'gradient_boosting': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge user config with defaults
                    for key, value in user_config.items():
                        if key in default_config and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading configuration: {str(e)}")
        
        return default_config
    
    def _load_models(self):
        """
        Load pre-trained models if available.
        """
        model_types = ['random_forest', 'gradient_boosting']
        
        for model_type in model_types:
            model_path = os.path.join(self.model_dir, f"{model_type}_model.joblib")
            scaler_path = os.path.join(self.model_dir, f"{model_type}_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[model_type] = joblib.load(model_path)
                    self.scalers[model_type] = joblib.load(scaler_path)
                    self.logger.info(f"Loaded pre-trained {model_type} model")
                except Exception as e:
                    self.logger.error(f"Error loading {model_type} model: {str(e)}")
    
    def predict_price_targets(self, features, entry_price, option_data, position_type='long'):
        """
        Predict price targets for an options position.
        
        Args:
            features (dict): Features for price target prediction
            entry_price (float): Entry price of the position
            option_data (dict): Option contract data
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            list: List of price targets with percentage of position to exit
        """
        self.logger.info(f"Predicting price targets for {option_data.get('symbol', 'unknown')}")
        
        # Get configuration values
        profit_taking_levels = self.config['profit_taking_levels']
        position_sizing = self.config['position_sizing']
        adjust_for_volatility = self.config['adjust_for_volatility']
        volatility_adjustment_factor = self.config['volatility_adjustment_factor']
        min_profit_target = self.config['min_profit_target']
        max_profit_target = self.config['max_profit_target']
        
        # Get option features
        option_features = features['option_features']
        implied_volatility = option_features['implied_volatility']
        days_to_expiration = option_features['days_to_expiration']
        
        # Try ML prediction first if models are available
        ml_price_targets = None
        
        if self.models:
            try:
                # Prepare features for ML model
                ml_features = self._prepare_ml_features(features, option_data)
                
                # Make predictions with each model
                predictions = {}
                for model_type, model in self.models.items():
                    if model_type in self.scalers:
                        # Scale features
                        scaled_features = self.scalers[model_type].transform([ml_features])
                        
                        # Predict profit target multiplier
                        prediction = model.predict(scaled_features)[0]
                        predictions[model_type] = prediction
                
                # Average predictions if we have multiple models
                if predictions:
                    # Calculate average profit target multiplier
                    avg_profit_multiplier = sum(predictions.values()) / len(predictions)
                    
                    # Ensure within bounds
                    avg_profit_multiplier = max(min_profit_target, min(avg_profit_multiplier, max_profit_target))
                    
                    # Generate price targets based on ML prediction
                    ml_price_targets = self._generate_price_targets_from_multiplier(
                        avg_profit_multiplier, entry_price, position_type, position_sizing
                    )
            except Exception as e:
                self.logger.error(f"Error in ML prediction for price targets: {str(e)}")
                ml_price_targets = None
        
        # If ML prediction is available, use it
        if ml_price_targets:
            price_targets = ml_price_targets
        else:
            # Use rule-based approach as fallback
            price_targets = self._rule_based_price_targets(
                option_features, entry_price, position_type, features
            )
        
        self.logger.info(f"Generated {len(price_targets)} price targets")
        return price_targets
    
    def _rule_based_price_targets(self, option_features, entry_price, position_type, features):
        """
        Apply rule-based approach to determine price targets.
        
        Args:
            option_features (dict): Option-specific features
            entry_price (float): Entry price of the position
            position_type (str): Type of position ('long' or 'short')
            features (dict): All features for context
            
        Returns:
            list: List of price targets with percentage of position to exit
        """
        # Get configuration values
        profit_taking_levels = self.config['profit_taking_levels']
        position_sizing = self.config['position_sizing']
        adjust_for_volatility = self.config['adjust_for_volatility']
        volatility_adjustment_factor = self.config['volatility_adjustment_factor']
        min_profit_target = self.config['min_profit_target']
        max_profit_target = self.config['max_profit_target']
        
        # Extract needed features
        implied_volatility = option_features['implied_volatility']
        days_to_expiration = option_features['days_to_expiration']
        delta = abs(option_features['delta'])
        
        # Initialize price targets
        price_targets = []
        
        # Adjust profit taking levels based on implied volatility if configured
        adjusted_levels = profit_taking_levels.copy()
        
        if adjust_for_volatility:
            # Higher volatility = higher potential profit targets
            volatility_factor = 1.0 + (implied_volatility - 0.3) * volatility_adjustment_factor
            volatility_factor = max(0.5, min(volatility_factor, 2.0))  # Limit adjustment range
            
            adjusted_levels = [level * volatility_factor for level in profit_taking_levels]
        
        # Adjust based on days to expiration
        # Shorter time to expiration = lower profit targets
        if days_to_expiration < 14:  # Two weeks or less
            time_factor = max(0.5, days_to_expiration / 14)  # Scale down for short-dated options
            adjusted_levels = [level * time_factor for level in adjusted_levels]
        
        # Adjust based on delta (probability of profit)
        # Higher delta (deeper ITM) = lower profit targets
        # Lower delta (deeper OTM) = higher profit targets (but lower probability)
        if delta > 0.7:  # Deep ITM
            delta_factor = 0.8  # Reduce profit targets
        elif delta < 0.3:  # Deep OTM
            delta_factor = 1.2  # Increase profit targets
        else:
            delta_factor = 1.0  # No adjustment
        
        adjusted_levels = [level * delta_factor for level in adjusted_levels]
        
        # Ensure profit targets are within bounds
        adjusted_levels = [max(min_profit_target, min(level, max_profit_target)) for level in adjusted_levels]
        
        # Calculate price targets based on entry price
        for i, level in enumerate(adjusted_levels):
            if position_type == 'long':
                # For long positions, price targets are above entry price
                target_price = entry_price * (1 + level)
            else:
                # For short positions, price targets are below entry price
                target_price = entry_price * (1 - level)
            
            # Add price target with position sizing
            price_targets.append({
                'price': target_price,
                'percentage': position_sizing[i] if i < len(position_sizing) else 0.25,
                'profit_percentage': level * 100
            })
        
        return price_targets
    
    def _generate_price_targets_from_multiplier(self, profit_multiplier, entry_price, position_type, position_sizing):
        """
        Generate price targets from a profit multiplier.
        
        Args:
            profit_multiplier (float): Profit target multiplier
            entry_price (float): Entry price of the position
            position_type (str): Type of position ('long' or 'short')
            position_sizing (list): List of position sizing percentages
            
        Returns:
            list: List of price targets with percentage of position to exit
        """
        # Calculate intermediate levels based on the final profit target
        levels = [profit_multiplier * ratio for ratio in [0.25, 0.5, 0.75, 1.0]]
        
        # Calculate price targets
        price_targets = []
        
        for i, level in enumerate(levels):
            if position_type == 'long':
                # For long positions, price targets are above entry price
                target_price = entry_price * (1 + level)
            else:
                # For short positions, price targets are below entry price
                target_price = entry_price * (1 - level)
            
            # Add price target with position sizing
            price_targets.append({
                'price': target_price,
                'percentage': position_sizing[i] if i < len(position_sizing) else 0.25,
                'profit_percentage': level * 100
            })
        
        return price_targets
    
    def _prepare_ml_features(self, features, option_data):
        """
        Prepare features for ML model.
        
        Args:
            features (dict): Features dictionary
            option_data (dict): Option contract data
            
        Returns:
            list: Feature vector for ML model
        """
        # Get option features
        option_features = features['option_features']
        
        # Calculate moneyness
        underlying_price = option_data.get('underlyingPrice', 0)
        strike_price = option_data.get('strike', 0)
        option_type = option_data.get('option_type', '')
        
        if underlying_price > 0 and strike_price > 0:
            if option_type.upper() == 'CALL':
                moneyness = underlying_price / strike_price - 1
            else:  # PUT
                moneyness = strike_price / underlying_price - 1
        else:
            moneyness = 0
        
        # Calculate volume/open interest ratio
        volume = option_data.get('volume', 0)
        open_interest = option_data.get('open_interest', 1)  # Avoid division by zero
        volume_oi_ratio = volume / open_interest if open_interest > 0 else 0
        
        # Get market data if available
        market_data = features.get('market_data', {})
        market_context = market_data.get('market_context', {})
        indicators = market_data.get('indicators', {})
        
        # Extract market features
        trend_strength = market_context.get('market_trend', 0)
        volatility_regime = market_context.get('volatility_regime', 0)
        rsi = indicators.get('rsi', 50)
        macd_histogram = indicators.get('macd_histogram', 0)
        
        # Create feature vector based on config
        feature_vector = []
        
        # Add option features
        for feature_name in self.config['model_features']:
            if feature_name == 'days_to_expiration':
                feature_vector.append(option_features.get('days_to_expiration', 30))
            elif feature_name == 'implied_volatility':
                feature_vector.append(option_features.get('implied_volatility', 0.3))
            elif feature_name == 'delta':
                feature_vector.append(option_features.get('delta', 0.5))
            elif feature_name == 'gamma':
                feature_vector.append(option_features.get('gamma', 0.05))
            elif feature_name == 'theta':
                feature_vector.append(option_features.get('theta', -0.05))
            elif feature_name == 'vega':
                feature_vector.append(option_features.get('vega', 0.1))
            elif feature_name == 'rho':
                feature_vector.append(option_features.get('rho', 0.01))
            elif feature_name == 'option_type':
                feature_vector.append(1 if option_type.upper() == 'CALL' else 0)
            elif feature_name == 'moneyness':
                feature_vector.append(moneyness)
            elif feature_name == 'volume_oi_ratio':
                feature_vector.append(volume_oi_ratio)
        
        # Add market features
        for feature_name in self.config['market_features']:
            if feature_name == 'trend_strength':
                feature_vector.append(trend_strength)
            elif feature_name == 'volatility_regime':
                feature_vector.append(volatility_regime)
            elif feature_name == 'rsi':
                feature_vector.append(rsi)
            elif feature_name == 'macd_histogram':
                feature_vector.append(macd_histogram)
        
        return feature_vector
    
    def calculate_stop_loss_take_profit(self, price_targets, entry_price, option_data, position_type='long'):
        """
        Calculate stop loss and take profit levels.
        
        Args:
            price_targets (list): List of price targets
            entry_price (float): Entry price of the position
            option_data (dict): Option contract data
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            tuple: (stop_loss, take_profit) prices
        """
        self.logger.info("Calculating stop loss and take profit levels")
        
        # Get configuration values
        risk_reward_ratio = self.config['risk_reward_ratio']
        
        # Get option features
        implied_volatility = option_data.get('implied_volatility', 0.3)
        
        # Calculate take profit as the highest price target
        if price_targets:
            if position_type == 'long':
                take_profit = max([target['price'] for target in price_targets])
            else:
                take_profit = min([target['price'] for target in price_targets])
        else:
            # Fallback if no price targets
            if position_type == 'long':
                take_profit = entry_price * (1 + implied_volatility)
            else:
                take_profit = entry_price * (1 - implied_volatility)
        
        # Calculate stop loss based on risk-reward ratio
        if position_type == 'long':
            # For long positions, calculate how far below entry price
            profit_amount = take_profit - entry_price
            risk_amount = profit_amount / risk_reward_ratio
            stop_loss = max(entry_price - risk_amount, entry_price * 0.7)  # Limit to 30% loss
        else:
            # For short positions, calculate how far above entry price
            profit_amount = entry_price - take_profit
            risk_amount = profit_amount / risk_reward_ratio
            stop_loss = min(entry_price + risk_amount, entry_price * 1.3)  # Limit to 30% loss
        
        return stop_loss, take_profit
    
    def train_model(self, training_data, target_column='profit_multiplier'):
        """
        Train ML models for price target prediction.
        
        Args:
            training_data (pandas.DataFrame): Training data with features and target
            target_column (str): Name of the target column
            
        Returns:
            dict: Training results and metrics
        """
        self.logger.info(f"Training price target models on {len(training_data)} samples")
        
        try:
            # Prepare features and target
            X = training_data.drop(target_column, axis=1)
            y = training_data[target_column]
            
            # Train models
            results = {}
            
            # Random Forest model
            rf_config = self.config['models']['random_forest']
            rf_model = RandomForestRegressor(
                n_estimators=rf_config.get('n_estimators', 100),
                max_depth=rf_config.get('max_depth', 10),
                min_samples_split=rf_config.get('min_samples_split', 5),
                min_samples_leaf=rf_config.get('min_samples_leaf', 2),
                random_state=42
            )
            
            # Scale features
            rf_scaler = StandardScaler()
            X_scaled = rf_scaler.fit_transform(X)
            
            # Train model
            rf_model.fit(X_scaled, y)
            
            # Save model and scaler
            joblib.dump(rf_model, os.path.join(self.model_dir, "random_forest_model.joblib"))
            joblib.dump(rf_scaler, os.path.join(self.model_dir, "random_forest_scaler.joblib"))
            
            # Store in instance
            self.models['random_forest'] = rf_model
            self.scalers['random_forest'] = rf_scaler
            
            # Calculate metrics
            rf_predictions = rf_model.predict(X_scaled)
            rf_mse = np.mean((rf_predictions - y) ** 2)
            rf_mae = np.mean(np.abs(rf_predictions - y))
            rf_r2 = rf_model.score(X_scaled, y)
            
            results['random_forest'] = {
                'mse': rf_mse,
                'mae': rf_mae,
                'r2': rf_r2,
                'feature_importance': dict(zip(X.columns, rf_model.feature_importances_))
            }
            
            # Gradient Boosting model
            gb_config = self.config['models']['gradient_boosting']
            gb_model = GradientBoostingRegressor(
                n_estimators=gb_config.get('n_estimators', 100),
                learning_rate=gb_config.get('learning_rate', 0.1),
                max_depth=gb_config.get('max_depth', 5),
                min_samples_split=gb_config.get('min_samples_split', 5),
                min_samples_leaf=gb_config.get('min_samples_leaf', 2),
                random_state=42
            )
            
            # Scale features
            gb_scaler = StandardScaler()
            X_scaled = gb_scaler.fit_transform(X)
            
            # Train model
            gb_model.fit(X_scaled, y)
            
            # Save model and scaler
            joblib.dump(gb_model, os.path.join(self.model_dir, "gradient_boosting_model.joblib"))
            joblib.dump(gb_scaler, os.path.join(self.model_dir, "gradient_boosting_scaler.joblib"))
            
            # Store in instance
            self.models['gradient_boosting'] = gb_model
            self.scalers['gradient_boosting'] = gb_scaler
            
            # Calculate metrics
            gb_predictions = gb_model.predict(X_scaled)
            gb_mse = np.mean((gb_predictions - y) ** 2)
            gb_mae = np.mean(np.abs(gb_predictions - y))
            gb_r2 = gb_model.score(X_scaled, y)
            
            results['gradient_boosting'] = {
                'mse': gb_mse,
                'mae': gb_mae,
                'r2': gb_r2,
                'feature_importance': dict(zip(X.columns, gb_model.feature_importances_))
            }
            
            self.logger.info(f"Successfully trained price target models")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training price target models: {str(e)}")
            raise
