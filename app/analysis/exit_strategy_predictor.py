"""
Exit Strategy Predictor Module

This module provides functionality for predicting optimal exit timing and price targets
for options positions. It integrates with the existing recommendation engine to provide
complete entry and exit recommendations for options trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json

# Import ML components
from app.ml.enhanced.integration import EnhancedMLIntegration
from app.ml.enhanced.feature_manager import FeatureEngineeringManager
from app.indicators.technical_indicators import TechnicalIndicators
from app.indicators.multi_timeframe_analyzer import MultiTimeframeAnalyzer

# Import utils
from app.utils.enhanced_logging import EnhancedLogger

class ExitStrategyPredictor:
    """
    Predicts optimal exit timing and price targets for options positions.
    
    This class analyzes options data, market conditions, and technical indicators
    to generate predictions for when to exit options positions and at what price
    targets to maximize profits while managing risk.
    """
    
    def __init__(self, ml_integration=None, data_collector=None, config_path=None):
        """
        Initialize the exit strategy predictor.
        
        Args:
            ml_integration: ML integration instance for predictions
            data_collector: Data collector for market data
            config_path: Path to configuration file
        """
        self.logger = EnhancedLogger('exit_strategy_predictor')
        self.logger.info("Initializing Exit Strategy Predictor")
        
        # Initialize ML integration if not provided
        if ml_integration is None:
            self.logger.info("Creating new ML integration instance")
            self.ml_integration = EnhancedMLIntegration(config_path=config_path)
        else:
            self.ml_integration = ml_integration
            
        # Store data collector
        self.data_collector = data_collector
        
        # Initialize technical indicators
        self.technical_indicators = TechnicalIndicators()
        
        # Initialize multi-timeframe analyzer
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize feature engineering manager
        self.feature_manager = FeatureEngineeringManager()
        
        # Create cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'exit_strategies')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger.info("Exit Strategy Predictor initialized successfully")
    
    def _load_config(self, config_path=None):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            'exit_timing': {
                'min_holding_days': 1,
                'max_holding_days': 30,
                'time_decay_threshold': 0.03,
                'profit_target_multiplier': 2.0,
                'stop_loss_multiplier': 1.0
            },
            'price_targets': {
                'profit_taking_levels': [0.25, 0.5, 0.75, 1.0],
                'position_sizing': [0.25, 0.25, 0.25, 0.25],
                'adjust_for_volatility': True
            },
            'market_conditions': {
                'high_volatility_threshold': 25,
                'low_volatility_threshold': 15,
                'trend_strength_threshold': 0.6
            },
            'ml_models': {
                'use_ensemble': True,
                'time_series_lookback': 20,
                'feature_importance_threshold': 0.05
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
    
    def predict_exit_strategy(self, option_data, entry_price, entry_date=None, position_type='long'):
        """
        Predict optimal exit strategy for an options position.
        
        Args:
            option_data (dict): Option contract data
            entry_price (float): Entry price of the position
            entry_date (datetime, optional): Entry date of the position
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            dict: Exit strategy including:
                - optimal_exit_time: Predicted optimal exit time
                - price_targets: List of price targets for partial exits
                - stop_loss: Recommended stop loss price
                - take_profit: Recommended take profit price
                - exit_probability: Probability of successful exit
                - days_to_hold: Recommended holding period in days
        """
        self.logger.info(f"Predicting exit strategy for {option_data.get('symbol', 'unknown')}")
        
        # Set entry date to now if not provided
        if entry_date is None:
            entry_date = datetime.now()
        
        # Extract features for exit prediction
        features = self._extract_exit_features(option_data, entry_price, entry_date, position_type)
        
        # Predict optimal exit time
        exit_timing = self._predict_exit_timing(features, option_data, entry_date)
        
        # Predict price targets
        price_targets = self._predict_price_targets(features, entry_price, option_data, position_type)
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_exit_points(
            features, entry_price, price_targets, option_data, position_type
        )
        
        # Calculate exit probability
        exit_probability = self._calculate_exit_probability(features, price_targets, option_data)
        
        # Create exit strategy
        exit_strategy = {
            'symbol': option_data.get('symbol', ''),
            'underlying': option_data.get('underlying', ''),
            'option_type': option_data.get('option_type', ''),
            'strike': option_data.get('strike', 0),
            'expiration_date': option_data.get('expiration_date', ''),
            'entry_price': entry_price,
            'entry_date': entry_date.isoformat() if isinstance(entry_date, datetime) else entry_date,
            'optimal_exit_time': exit_timing['optimal_exit_time'].isoformat() if isinstance(exit_timing['optimal_exit_time'], datetime) else exit_timing['optimal_exit_time'],
            'days_to_hold': exit_timing['days_to_hold'],
            'price_targets': price_targets,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'exit_probability': exit_probability,
            'position_type': position_type,
            'exit_reasons': exit_timing['exit_reasons'],
            'confidence_score': exit_timing['confidence_score']
        }
        
        self.logger.info(f"Exit strategy prediction completed for {option_data.get('symbol', 'unknown')}")
        return exit_strategy
    
    def _extract_exit_features(self, option_data, entry_price, entry_date, position_type):
        """
        Extract features relevant for exit prediction.
        
        Args:
            option_data (dict): Option contract data
            entry_price (float): Entry price of the position
            entry_date (datetime): Entry date of the position
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            dict: Features for exit prediction
        """
        self.logger.info("Extracting features for exit prediction")
        
        # Get underlying symbol
        underlying_symbol = option_data.get('underlying', '')
        
        # Get market data if data collector is available
        market_data = {}
        if self.data_collector:
            try:
                # Get underlying price data
                price_data = self.data_collector.get_price_data(underlying_symbol, lookback_days=30)
                
                # Get technical indicators
                indicators = self.technical_indicators.calculate_all_indicators(price_data)
                
                # Get multi-timeframe analysis
                multi_timeframe_data = self.multi_timeframe_analyzer.analyze(underlying_symbol)
                
                # Get market context
                market_context = self._get_market_context()
                
                market_data = {
                    'price_data': price_data,
                    'indicators': indicators,
                    'multi_timeframe_data': multi_timeframe_data,
                    'market_context': market_context
                }
            except Exception as e:
                self.logger.error(f"Error getting market data: {str(e)}")
        
        # Calculate time to expiration
        days_to_expiration = 30  # Default value
        try:
            if 'expiration_date' in option_data:
                expiration_date = option_data['expiration_date']
                if isinstance(expiration_date, str):
                    expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
                days_to_expiration = (expiration_date - entry_date).days if isinstance(entry_date, datetime) else 30
        except Exception as e:
            self.logger.error(f"Error calculating days to expiration: {str(e)}")
        
        # Extract option-specific features
        option_features = {
            'strike': option_data.get('strike', 0),
            'option_type': 1 if option_data.get('option_type', '') == 'CALL' else 0,
            'days_to_expiration': days_to_expiration,
            'implied_volatility': option_data.get('implied_volatility', 0.3),
            'delta': option_data.get('delta', 0.5),
            'gamma': option_data.get('gamma', 0.05),
            'theta': option_data.get('theta', -0.05),
            'vega': option_data.get('vega', 0.1),
            'rho': option_data.get('rho', 0.01),
            'bid': option_data.get('bid', 0),
            'ask': option_data.get('ask', 0),
            'last': option_data.get('last', 0),
            'volume': option_data.get('volume', 0),
            'open_interest': option_data.get('open_interest', 0),
            'entry_price': entry_price,
            'position_type': 1 if position_type == 'long' else 0
        }
        
        # Combine all features
        features = {
            'option_features': option_features,
            'market_data': market_data,
            'underlying_symbol': underlying_symbol,
            'entry_date': entry_date
        }
        
        return features
    
    def _predict_exit_timing(self, features, option_data, entry_date):
        """
        Predict optimal time to exit position.
        
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
        self.logger.info("Predicting optimal exit timing")
        
        # Get configuration values
        config = self.config['exit_timing']
        min_holding_days = config['min_holding_days']
        max_holding_days = config['max_holding_days']
        
        # Get option features
        option_features = features['option_features']
        days_to_expiration = option_features['days_to_expiration']
        theta = option_features['theta']
        implied_volatility = option_features['implied_volatility']
        
        # Initialize variables
        days_to_hold = min_holding_days
        exit_reasons = []
        confidence_score = 0.5
        
        # Try to use ML prediction if available
        ml_prediction = None
        try:
            if hasattr(self, 'ml_integration') and self.ml_integration:
                ml_prediction = self.ml_integration.predict(features)
                if ml_prediction and 'exit_timing' in ml_prediction:
                    days_to_hold = ml_prediction['exit_timing'].get('days_to_hold', days_to_hold)
                    confidence_score = ml_prediction['exit_timing'].get('confidence', confidence_score)
                    exit_reasons.append(f"ML model prediction (confidence: {confidence_score:.2f})")
        except Exception as e:
            self.logger.error(f"Error in ML prediction for exit timing: {str(e)}")
        
        # If no ML prediction, use rule-based approach
        if ml_prediction is None or 'exit_timing' not in ml_prediction:
            # Calculate based on time decay (theta)
            if theta < -config['time_decay_threshold']:
                # High time decay, exit sooner
                theta_based_days = min(days_to_expiration * 0.3, max_holding_days)
                exit_reasons.append(f"High time decay (theta: {theta:.4f})")
            else:
                # Lower time decay, can hold longer
                theta_based_days = min(days_to_expiration * 0.5, max_holding_days)
                exit_reasons.append(f"Moderate time decay (theta: {theta:.4f})")
            
            # Adjust based on implied volatility
            if implied_volatility > 0.5:
                # High volatility, exit sooner
                volatility_adjustment = 0.8
                exit_reasons.append(f"High implied volatility ({implied_volatility:.2f})")
            elif implied_volatility < 0.2:
                # Low volatility, can hold longer
                volatility_adjustment = 1.2
                exit_reasons.append(f"Low implied volatility ({implied_volatility:.2f})")
            else:
                # Moderate volatility
                volatility_adjustment = 1.0
                exit_reasons.append(f"Moderate implied volatility ({implied_volatility:.2f})")
            
            # Apply adjustments
            days_to_hold = max(min_holding_days, min(theta_based_days * volatility_adjustment, max_holding_days))
            
            # Ensure we don't exceed expiration
            days_to_hold = min(days_to_hold, days_to_expiration - 1) if days_to_expiration > 1 else 1
            
            # Calculate confidence score based on rule certainty
            confidence_score = 0.6  # Base confidence for rule-based approach
        
        # Calculate optimal exit time
        optimal_exit_time = entry_date + timedelta(days=days_to_hold) if isinstance(entry_date, datetime) else None
        
        # Ensure we don't exceed expiration date
        if isinstance(optimal_exit_time, datetime) and 'expiration_date' in option_data:
            expiration_date = option_data['expiration_date']
            if isinstance(expiration_date, str):
                expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
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
    
    def _predict_price_targets(self, features, entry_price, option_data, position_type):
        """
        Predict price targets for partial exits.
        
        Args:
            features (dict): Features for exit prediction
            entry_price (float): Entry price of the position
            option_data (dict): Option contract data
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            list: List of price targets with percentage of position to exit
        """
        self.logger.info("Predicting price targets")
        
        # Get configuration values
        config = self.config['price_targets']
        profit_taking_levels = config['profit_taking_levels']
        position_sizing = config['position_sizing']
        
        # Get option features
        option_features = features['option_features']
        implied_volatility = option_features['implied_volatility']
        
        # Initialize price targets
        price_targets = []
        
        # Try to use ML prediction if available
        ml_prediction = None
        try:
            if hasattr(self, 'ml_integration') and self.ml_integration:
                ml_prediction = self.ml_integration.predict(features)
                if ml_prediction and 'price_targets' in ml_prediction:
                    return ml_prediction['price_targets']
        except Exception as e:
            self.logger.error(f"Error in ML prediction for price targets: {str(e)}")
        
        # If no ML prediction, use rule-based approach
        if ml_prediction is None or 'price_targets' not in ml_prediction:
            # Adjust profit taking levels based on implied volatility if configured
            adjusted_levels = profit_taking_levels.copy()
            if config['adjust_for_volatility']:
                # Higher volatility = higher potential profit targets
                volatility_factor = 1.0 + (implied_volatility - 0.3) * 2  # Normalize around 0.3 IV
                volatility_factor = max(0.5, min(volatility_factor, 2.0))  # Limit adjustment range
                
                adjusted_levels = [level * volatility_factor for level in profit_taking_levels]
            
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
    
    def _calculate_exit_points(self, features, entry_price, price_targets, option_data, position_type):
        """
        Calculate stop loss and take profit levels.
        
        Args:
            features (dict): Features for exit prediction
            entry_price (float): Entry price of the position
            price_targets (list): List of price targets
            option_data (dict): Option contract data
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            tuple: (stop_loss, take_profit) prices
        """
        self.logger.info("Calculating stop loss and take profit levels")
        
        # Get configuration values
        config = self.config['exit_timing']
        stop_loss_multiplier = config['stop_loss_multiplier']
        profit_target_multiplier = config['profit_target_multiplier']
        
        # Get option features
        option_features = features['option_features']
        implied_volatility = option_features['implied_volatility']
        
        # Calculate stop loss based on implied volatility and entry price
        if position_type == 'long':
            # For long positions, stop loss is below entry price
            volatility_based_stop = entry_price * (1 - implied_volatility * stop_loss_multiplier)
            stop_loss = max(volatility_based_stop, entry_price * 0.7)  # Limit to 30% loss
        else:
            # For short positions, stop loss is above entry price
            volatility_based_stop = entry_price * (1 + implied_volatility * stop_loss_multiplier)
            stop_loss = min(volatility_based_stop, entry_price * 1.3)  # Limit to 30% loss
        
        # Calculate take profit as the highest price target
        if price_targets:
            if position_type == 'long':
                take_profit = max([target['price'] for target in price_targets])
            else:
                take_profit = min([target['price'] for target in price_targets])
        else:
            # Fallback if no price targets
            if position_type == 'long':
                take_profit = entry_price * (1 + implied_volatility * profit_target_multiplier)
            else:
                take_profit = entry_price * (1 - implied_volatility * profit_target_multiplier)
        
        return stop_loss, take_profit
    
    def _calculate_exit_probability(self, features, price_targets, option_data):
        """
        Calculate probability of successful exit.
        
        Args:
            features (dict): Features for exit prediction
            price_targets (list): List of price targets
            option_data (dict): Option contract data
            
        Returns:
            float: Probability of successful exit (0-1)
        """
        self.logger.info("Calculating exit probability")
        
        # Get option features
        option_features = features['option_features']
        delta = abs(option_features['delta'])
        
        # Try to use ML prediction if available
        try:
            if hasattr(self, 'ml_integration') and self.ml_integration:
                ml_prediction = self.ml_integration.predict(features)
                if ml_prediction and 'exit_probability' in ml_prediction:
                    return ml_prediction['exit_probability']
        except Exception as e:
            self.logger.error(f"Error in ML prediction for exit probability: {str(e)}")
        
        # If no ML prediction, use delta as a base probability
        # Delta is often interpreted as approximate probability of option expiring ITM
        base_probability = delta
        
        # Adjust based on market conditions if available
        market_data = features.get('market_data', {})
        market_context = market_data.get('market_context', {})
        
        if market_context:
            # Adjust based on market trend alignment
            trend_alignment = market_context.get('trend_alignment', 0)
            trend_adjustment = 0.1 * trend_alignment  # -0.1 to +0.1 adjustment
            
            # Adjust based on volatility regime
            volatility_regime = market_context.get('volatility_regime', 0)
            volatility_adjustment = 0.05 * volatility_regime  # -0.05 to +0.05 adjustment
            
            # Apply adjustments
            adjusted_probability = base_probability + trend_adjustment + volatility_adjustment
            
            # Ensure probability is between 0 and 1
            exit_probability = max(0.1, min(adjusted_probability, 0.9))
        else:
            # Without market context, use base probability with slight discount
            exit_probability = max(0.1, min(base_probability * 0.9, 0.9))
        
        return exit_probability
    
    def _get_market_context(self):
        """
        Get current market context for decision making.
        
        Returns:
            dict: Market context information
        """
        # This would typically involve getting VIX, market trend, sector performance, etc.
        # For now, return a simplified context
        return {
            'volatility_regime': 0,  # -1 (low), 0 (normal), 1 (high)
            'market_trend': 0,       # -1 (down), 0 (sideways), 1 (up)
            'trend_alignment': 0,    # -1 (against), 0 (neutral), 1 (aligned)
            'liquidity': 0           # -1 (low), 0 (normal), 1 (high)
        }
    
    def generate_exit_strategies_for_recommendations(self, recommendations):
        """
        Generate exit strategies for a list of option recommendations.
        
        Args:
            recommendations (list): List of option recommendations
            
        Returns:
            list: Enhanced recommendations with exit strategies
        """
        self.logger.info(f"Generating exit strategies for {len(recommendations)} recommendations")
        
        enhanced_recommendations = []
        
        for rec in recommendations:
            try:
                # Extract option data
                option_data = {
                    'symbol': rec.get('symbol', ''),
                    'underlying': rec.get('underlying', ''),
                    'option_type': rec.get('option_type', 'CALL'),
                    'strike': rec.get('strike', 0),
                    'expiration_date': rec.get('expiration_date', ''),
                    'bid': rec.get('bid', 0),
                    'ask': rec.get('ask', 0),
                    'last': rec.get('last', 0),
                    'volume': rec.get('volume', 0),
                    'open_interest': rec.get('open_interest', 0),
                    'delta': rec.get('delta', 0.5),
                    'gamma': rec.get('gamma', 0.05),
                    'theta': rec.get('theta', -0.05),
                    'vega': rec.get('vega', 0.1),
                    'rho': rec.get('rho', 0.01),
                    'implied_volatility': rec.get('implied_volatility', 0.3)
                }
                
                # Get entry price (mid price if available, otherwise use last)
                entry_price = rec.get('price', 0)
                if entry_price == 0:
                    bid = rec.get('bid', 0)
                    ask = rec.get('ask', 0)
                    if bid > 0 and ask > 0:
                        entry_price = (bid + ask) / 2
                    else:
                        entry_price = rec.get('last', 1)
                
                # Determine position type (long for both calls and puts in this case)
                position_type = 'long'
                
                # Generate exit strategy
                exit_strategy = self.predict_exit_strategy(
                    option_data, entry_price, datetime.now(), position_type
                )
                
                # Enhance recommendation with exit strategy
                enhanced_rec = rec.copy()
                enhanced_rec.update({
                    'exitStrategy': {
                        'optimalExitDate': exit_strategy['optimal_exit_time'],
                        'daysToHold': exit_strategy['days_to_hold'],
                        'priceTargets': exit_strategy['price_targets'],
                        'stopLoss': exit_strategy['stop_loss'],
                        'takeProfit': exit_strategy['take_profit'],
                        'exitProbability': exit_strategy['exit_probability'],
                        'exitReasons': exit_strategy['exit_reasons'],
                        'confidenceScore': exit_strategy['confidence_score']
                    }
                })
                
                enhanced_recommendations.append(enhanced_rec)
                
            except Exception as e:
                self.logger.error(f"Error generating exit strategy for recommendation: {str(e)}")
                enhanced_recommendations.append(rec)
        
        self.logger.info(f"Generated exit strategies for {len(enhanced_recommendations)} recommendations")
        return enhanced_recommendations
