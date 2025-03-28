"""
Exit Timing Prediction Module

This module provides specialized functionality for predicting optimal exit timing
for options positions based on various factors including time decay, volatility,
and market conditions.
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

class ExitTimingPredictor:
    """
    Specialized predictor for optimal exit timing of options positions.
    
    This class focuses specifically on the timing aspect of exit strategies,
    using both rule-based approaches and machine learning to determine the
    optimal time to exit an options position.
    """
    
    def __init__(self, data_collector=None, config_path=None):
        """
        Initialize the exit timing predictor.
        
        Args:
            data_collector: Data collector for market data
            config_path: Path to configuration file
        """
        self.logger = EnhancedLogger('exit_timing_predictor')
        self.logger.info("Initializing Exit Timing Predictor")
        
        # Store data collector
        self.data_collector = data_collector
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        
        # Create model directory
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'exit_timing')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load pre-trained models if available
        self._load_models()
        
        self.logger.info("Exit Timing Predictor initialized successfully")
    
    def _load_config(self, config_path=None):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            'min_holding_days': 1,
            'max_holding_days': 30,
            'time_decay_threshold': 0.03,
            'high_volatility_threshold': 0.4,
            'low_volatility_threshold': 0.2,
            'expiration_buffer_days': 5,
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
    
    def predict_exit_timing(self, features, option_data, entry_date):
        """
        Predict optimal exit timing for an options position.
        
        Args:
            features (dict): Features for exit prediction
            option_data (dict): Option contract data
            entry_date (datetime): Entry date of the position
            
        Returns:
            dict: Exit timing prediction including:
                - optimal_exit_time: Predicted optimal exit time
                - days_to_hold: Recommended holding period in days
                - exit_reasons: Reasons for the exit recommendation
                - confidence_score: Confidence in the prediction
        """
        self.logger.info(f"Predicting exit timing for {option_data.get('symbol', 'unknown')}")
        
        # Get configuration values
        min_holding_days = self.config['min_holding_days']
        max_holding_days = self.config['max_holding_days']
        
        # Get option features
        option_features = features['option_features']
        days_to_expiration = option_features['days_to_expiration']
        theta = option_features['theta']
        implied_volatility = option_features['implied_volatility']
        delta = option_features['delta']
        gamma = option_features['gamma']
        
        # Calculate moneyness (how far in or out of the money)
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
        
        # Try ML prediction first if models are available
        ml_days_to_hold = None
        ml_confidence = 0
        exit_reasons = []
        
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
                        
                        # Predict days to hold
                        prediction = model.predict(scaled_features)[0]
                        predictions[model_type] = prediction
                
                # Average predictions if we have multiple models
                if predictions:
                    ml_days_to_hold = sum(predictions.values()) / len(predictions)
                    ml_confidence = 0.7  # Base confidence for ML prediction
                    
                    # Adjust confidence based on model agreement
                    if len(predictions) > 1:
                        # Calculate standard deviation of predictions
                        std_dev = np.std(list(predictions.values()))
                        # Higher agreement (lower std dev) = higher confidence
                        agreement_factor = max(0, 1 - std_dev / ml_days_to_hold if ml_days_to_hold > 0 else 0)
                        ml_confidence = min(0.9, ml_confidence + agreement_factor * 0.2)
                    
                    exit_reasons.append(f"ML model prediction (confidence: {ml_confidence:.2f})")
            except Exception as e:
                self.logger.error(f"Error in ML prediction for exit timing: {str(e)}")
                ml_days_to_hold = None
        
        # If ML prediction is available and confidence is high enough, use it
        if ml_days_to_hold is not None and ml_confidence >= 0.6:
            days_to_hold = max(min_holding_days, min(ml_days_to_hold, max_holding_days))
            confidence_score = ml_confidence
        else:
            # Use rule-based approach as fallback or enhancement
            rule_based_days, rule_reasons, rule_confidence = self._rule_based_exit_timing(
                option_features, moneyness, volume_oi_ratio, features
            )
            
            # If we have both ML and rule-based, blend them based on confidence
            if ml_days_to_hold is not None:
                # Weighted average based on confidence
                days_to_hold = (ml_days_to_hold * ml_confidence + rule_based_days * rule_confidence) / (ml_confidence + rule_confidence)
                confidence_score = max(ml_confidence, rule_confidence)
                exit_reasons.extend(rule_reasons)
            else:
                # Just use rule-based
                days_to_hold = rule_based_days
                confidence_score = rule_confidence
                exit_reasons = rule_reasons
        
        # Ensure days_to_hold is within bounds
        days_to_hold = max(min_holding_days, min(days_to_hold, max_holding_days))
        
        # Ensure we don't exceed expiration (with buffer)
        buffer_days = self.config['expiration_buffer_days']
        if days_to_expiration > buffer_days:
            days_to_hold = min(days_to_hold, days_to_expiration - buffer_days)
        else:
            days_to_hold = min(days_to_hold, max(1, days_to_expiration - 1))
            exit_reasons.append("Adjusted for near expiration")
        
        # Calculate optimal exit time
        optimal_exit_time = entry_date + timedelta(days=days_to_hold) if isinstance(entry_date, datetime) else None
        
        # Ensure we don't exceed expiration date
        if isinstance(optimal_exit_time, datetime) and 'expiration_date' in option_data:
            expiration_date = option_data['expiration_date']
            if isinstance(expiration_date, str):
                try:
                    expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
                except ValueError:
                    # Try alternative date formats
                    try:
                        expiration_date = datetime.strptime(expiration_date.split(':')[0], '%Y-%m-%d')
                    except:
                        expiration_date = entry_date + timedelta(days=days_to_expiration)
            
            if optimal_exit_time >= expiration_date:
                optimal_exit_time = expiration_date - timedelta(days=1)
                days_to_hold = (optimal_exit_time - entry_date).days if isinstance(entry_date, datetime) else days_to_hold
                exit_reasons.append("Adjusted to avoid expiration")
        
        return {
            'optimal_exit_time': optimal_exit_time,
            'days_to_hold': days_to_hold,
            'exit_reasons': exit_reasons,
            'confidence_score': confidence_score
        }
    
    def _rule_based_exit_timing(self, option_features, moneyness, volume_oi_ratio, features):
        """
        Apply rule-based approach to determine exit timing.
        
        Args:
            option_features (dict): Option-specific features
            moneyness (float): How far in/out of the money
            volume_oi_ratio (float): Volume to open interest ratio
            features (dict): All features for context
            
        Returns:
            tuple: (days_to_hold, exit_reasons, confidence_score)
        """
        # Get configuration values
        min_holding_days = self.config['min_holding_days']
        max_holding_days = self.config['max_holding_days']
        time_decay_threshold = self.config['time_decay_threshold']
        high_volatility_threshold = self.config['high_volatility_threshold']
        low_volatility_threshold = self.config['low_volatility_threshold']
        
        # Extract needed features
        days_to_expiration = option_features['days_to_expiration']
        theta = option_features['theta']
        implied_volatility = option_features['implied_volatility']
        delta = abs(option_features['delta'])
        gamma = option_features['gamma']
        
        # Initialize variables
        exit_reasons = []
        base_days = days_to_expiration * 0.4  # Start with 40% of time to expiration
        adjustments = []
        
        # 1. Time decay (theta) adjustment
        if theta < -time_decay_threshold:
            # High time decay, exit sooner
            theta_adjustment = -0.2  # Reduce holding time by 20%
            exit_reasons.append(f"High time decay (theta: {theta:.4f})")
        elif theta < -time_decay_threshold/2:
            # Moderate time decay
            theta_adjustment = -0.1  # Reduce holding time by 10%
            exit_reasons.append(f"Moderate time decay (theta: {theta:.4f})")
        else:
            # Low time decay, can hold longer
            theta_adjustment = 0.1  # Increase holding time by 10%
            exit_reasons.append(f"Low time decay (theta: {theta:.4f})")
        
        adjustments.append(theta_adjustment)
        
        # 2. Implied volatility adjustment
        if implied_volatility > high_volatility_threshold:
            # High volatility, exit sooner
            iv_adjustment = -0.15  # Reduce holding time by 15%
            exit_reasons.append(f"High implied volatility ({implied_volatility:.2f})")
        elif implied_volatility < low_volatility_threshold:
            # Low volatility, can hold longer
            iv_adjustment = 0.15  # Increase holding time by 15%
            exit_reasons.append(f"Low implied volatility ({implied_volatility:.2f})")
        else:
            # Moderate volatility
            iv_adjustment = 0
            exit_reasons.append(f"Moderate implied volatility ({implied_volatility:.2f})")
        
        adjustments.append(iv_adjustment)
        
        # 3. Moneyness adjustment
        if moneyness > 0.1:
            # Deep in the money, can hold longer
            moneyness_adjustment = 0.1  # Increase holding time by 10%
            exit_reasons.append(f"Deep in the money (moneyness: {moneyness:.2f})")
        elif moneyness < -0.1:
            # Deep out of the money, exit sooner
            moneyness_adjustment = -0.2  # Reduce holding time by 20%
            exit_reasons.append(f"Deep out of the money (moneyness: {moneyness:.2f})")
        else:
            # Near the money
            moneyness_adjustment = 0
            exit_reasons.append(f"Near the money (moneyness: {moneyness:.2f})")
        
        adjustments.append(moneyness_adjustment)
        
        # 4. Delta adjustment (sensitivity to underlying price)
        if delta > 0.7:
            # High delta, more sensitive to price changes
            delta_adjustment = 0.05  # Slight increase in holding time
            exit_reasons.append(f"High delta sensitivity ({delta:.2f})")
        elif delta < 0.3:
            # Low delta, less sensitive to price changes
            delta_adjustment = -0.1  # Reduce holding time
            exit_reasons.append(f"Low delta sensitivity ({delta:.2f})")
        else:
            # Moderate delta
            delta_adjustment = 0
            exit_reasons.append(f"Moderate delta sensitivity ({delta:.2f})")
        
        adjustments.append(delta_adjustment)
        
        # 5. Volume/OI ratio adjustment (liquidity indicator)
        if volume_oi_ratio > 1.0:
            # High trading activity, might indicate upcoming move
            vol_oi_adjustment = -0.1  # Reduce holding time
            exit_reasons.append(f"High trading activity (vol/OI: {volume_oi_ratio:.2f})")
        elif volume_oi_ratio < 0.1:
            # Low trading activity, might be harder to exit
            vol_oi_adjustment = -0.05  # Slight reduction in holding time
            exit_reasons.append(f"Low trading activity (vol/OI: {volume_oi_ratio:.2f})")
        else:
            # Moderate trading activity
            vol_oi_adjustment = 0
            exit_reasons.append(f"Moderate trading activity (vol/OI: {volume_oi_ratio:.2f})")
        
        adjustments.append(vol_oi_adjustment)
        
        # 6. Market context adjustment if available
        market_data = features.get('market_data', {})
        market_context = market_data.get('market_context', {})
        
        if market_context:
            # Adjust based on market trend
            market_trend = market_context.get('market_trend', 0)
            if market_trend > 0.5:
                # Strong uptrend
                trend_adjustment = 0.1  # Increase holding time
                exit_reasons.append("Strong market uptrend")
            elif market_trend < -0.5:
                # Strong downtrend
                trend_adjustment = -0.1  # Decrease holding time
                exit_reasons.append("Strong market downtrend")
            else:
                # Sideways market
                trend_adjustment = 0
                exit_reasons.append("Sideways market")
            
            adjustments.append(trend_adjustment)
            
            # Adjust based on volatility regime
            volatility_regime = market_context.get('volatility_regime', 0)
            if volatility_regime > 0.5:
                # High volatility regime
                vol_regime_adjustment = -0.1  # Decrease holding time
                exit_reasons.append("High market volatility regime")
            elif volatility_regime < -0.5:
                # Low volatility regime
                vol_regime_adjustment = 0.1  # Increase holding time
                exit_reasons.append("Low market volatility regime")
            else:
                # Normal volatility regime
                vol_regime_adjustment = 0
                exit_reasons.append("Normal market volatility regime")
            
            adjustments.append(vol_regime_adjustment)
        
        # Calculate net adjustment
        net_adjustment = 1.0 + sum(adjustments)
        
        # Apply adjustment to base days
        adjusted_days = base_days * net_adjustment
        
        # Ensure within bounds
        days_to_hold = max(min_holding_days, min(adjusted_days, max_holding_days))
        
        # Calculate confidence score based on rule certainty
        # More adjustments with stronger signals = higher confidence
        strong_signals = sum(1 for adj in adjustments if abs(adj) >= 0.1)
        confidence_score = 0.5 + (strong_signals / len(adjustments)) * 0.3  # 0.5 to 0.8 range
        
        return days_to_hold, exit_reasons, confidence_score
    
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
    
    def train_model(self, training_data, target_column='days_to_hold'):
        """
        Train ML models for exit timing prediction.
        
        Args:
            training_data (pandas.DataFrame): Training data with features and target
            target_column (str): Name of the target column
            
        Returns:
            dict: Training results and metrics
        """
        self.logger.info(f"Training exit timing models on {len(training_data)} samples")
        
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
            
            self.logger.info(f"Successfully trained exit timing models")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training exit timing models: {str(e)}")
            raise
