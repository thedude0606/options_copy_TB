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
            'confidence_score': exit_timing['confidence_score'],
            # Add detailed profit projections
            'profit_projections': self._calculate_profit_projections(entry_price, price_targets, option_data, position_type)
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
                multi_timeframe = self.multi_timeframe_analyzer.analyze(underlying_symbol, timeframes=['1d', '1w', '1m'])
                
                # Store market data
                market_data = {
                    'price_data': price_data,
                    'indicators': indicators,
                    'multi_timeframe': multi_timeframe
                }
            except Exception as e:
                self.logger.error(f"Error getting market data: {str(e)}")
        
        # Extract option features
        option_features = {
            'strike': option_data.get('strike', 0),
            'days_to_expiration': option_data.get('daysToExpiration', 30),
            'option_type': option_data.get('option_type', 'CALL'),
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
            'underlying_price': option_data.get('underlyingPrice', 0)
        }
        
        # Calculate time-based features
        time_features = {
            'days_since_entry': (datetime.now() - entry_date).days if isinstance(entry_date, datetime) else 0,
            'days_to_expiration': option_features['days_to_expiration'],
            'time_to_expiration_ratio': option_features['days_to_expiration'] / 30 if option_features['days_to_expiration'] > 0 else 0,
            'entry_day_of_week': entry_date.weekday() if isinstance(entry_date, datetime) else 0,
            'entry_month': entry_date.month if isinstance(entry_date, datetime) else 0
        }
        
        # Calculate position features
        position_features = {
            'entry_price': entry_price,
            'position_type': 1 if position_type == 'long' else 0,
            'current_price': option_features['last'] if option_features['last'] > 0 else entry_price,
            'price_change': (option_features['last'] - entry_price) / entry_price if option_features['last'] > 0 and entry_price > 0 else 0,
            'bid_ask_spread': (option_features['ask'] - option_features['bid']) / entry_price if option_features['ask'] > 0 and option_features['bid'] > 0 and entry_price > 0 else 0
        }
        
        # Combine all features
        features = {
            'option_features': option_features,
            'time_features': time_features,
            'position_features': position_features,
            'market_data': market_data
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
        
        # If no ML prediction, use enhanced rule-based approach
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
            
            # Enhanced: Consider delta for exit timing
            delta = option_features['delta']
            if abs(delta) > 0.7:
                # Deep in the money, can hold longer for intrinsic value
                days_to_hold = min(days_to_hold * 1.2, max_holding_days)
                exit_reasons.append(f"Deep in-the-money position (delta: {delta:.2f})")
            elif abs(delta) < 0.3:
                # Out of the money, exit sooner
                days_to_hold = max(min_holding_days, days_to_hold * 0.8)
                exit_reasons.append(f"Out-of-the-money position (delta: {delta:.2f})")
            
            # Enhanced: Consider gamma for exit timing
            gamma = option_features['gamma']
            if gamma > 0.1:
                # High gamma, more sensitive to price changes, monitor closely
                days_to_hold = max(min_holding_days, days_to_hold * 0.9)
                exit_reasons.append(f"High gamma sensitivity ({gamma:.2f})")
            
            # Calculate confidence score based on rule certainty
            confidence_score = 0.5 + (0.1 * len(exit_reasons))
            confidence_score = min(confidence_score, 0.9)  # Cap at 0.9 for rule-based
        
        # Calculate optimal exit time
        optimal_exit_time = entry_date + timedelta(days=days_to_hold) if isinstance(entry_date, datetime) else datetime.now() + timedelta(days=days_to_hold)
        
        # Return exit timing prediction
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
        delta = option_features['delta']
        days_to_expiration = option_features['days_to_expiration']
        
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
        
        # If no ML prediction, use enhanced rule-based approach
        if ml_prediction is None or 'price_targets' not in ml_prediction:
            # Enhanced: Adjust profit taking levels based on implied volatility and delta
            adjusted_levels = profit_taking_levels.copy()
            
            # Higher volatility = higher potential profit targets
            volatility_factor = 1.0 + (implied_volatility - 0.3) * 2  # Normalize around 0.3 IV
            volatility_factor = max(0.5, min(volatility_factor, 2.0))  # Limit adjustment range
            
            # Adjust based on delta (option moneyness)
            delta_factor = 1.0
            if abs(delta) > 0.7:
                # Deep in the money, more conservative targets
                delta_factor = 0.8
            elif abs(delta) < 0.3:
                # Out of the money, more aggressive targets
                delta_factor = 1.2
            
            # Adjust based on days to expiration
            dte_factor = 1.0
            if days_to_expiration < 7:
                # Short-dated options, more conservative targets
                dte_factor = 0.7
            elif days_to_expiration > 45:
                # Long-dated options, more aggressive targets
                dte_factor = 1.3
            
            # Apply all adjustments
            combined_factor = volatility_factor * delta_factor * dte_factor
            adjusted_levels = [level * combined_factor for level in profit_taking_levels]
            
            # Calculate price targets based on entry price
            for i, level in enumerate(adjusted_levels):
                if position_type == 'long':
                    # For long positions, price targets are above entry price
                    target_price = entry_price * (1 + level)
                    profit_percentage = level * 100
                else:
                    # For short positions, price targets are below entry price
                    target_price = entry_price * (1 - level)
                    profit_percentage = level * 100
                
                # Add price target with position sizing
                price_targets.append({
                    'price': target_price,
                    'percentage': position_sizing[i] if i < len(position_sizing) else 0.25,
                    'profit_percentage': profit_percentage,
                    'target_description': f"Target {i+1}: {profit_percentage:.1f}% profit"
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
            volatility_adjusted_stop = implied_volatility * entry_price * stop_loss_multiplier
            stop_loss = max(0.01, entry_price - volatility_adjusted_stop)
        else:
            # For short positions, stop loss is above entry price
            volatility_adjusted_stop = implied_volatility * entry_price * stop_loss_multiplier
            stop_loss = entry_price + volatility_adjusted_stop
        
        # Take profit is the highest price target
        if price_targets:
            if position_type == 'long':
                # For long positions, take highest price target
                take_profit = max([target['price'] for target in price_targets])
            else:
                # For short positions, take lowest price target
                take_profit = min([target['price'] for target in price_targets])
        else:
            # If no price targets, use profit target multiplier
            if position_type == 'long':
                take_profit = entry_price * (1 + profit_target_multiplier * implied_volatility)
            else:
                take_profit = entry_price * (1 - profit_target_multiplier * implied_volatility)
        
        # Enhanced: Ensure reasonable risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 1.0
        
        # Adjust if risk-reward ratio is too low
        if risk_reward_ratio < 1.5 and position_type == 'long':
            # Either adjust stop loss down or take profit up
            if risk > entry_price * 0.15:  # If stop loss is already more than 15% away
                take_profit = entry_price + (1.5 * risk)  # Adjust take profit up
            else:
                stop_loss = entry_price - (reward / 1.5)  # Adjust stop loss down
        elif risk_reward_ratio < 1.5 and position_type == 'short':
            # Either adjust stop loss up or take profit down
            if risk > entry_price * 0.15:  # If stop loss is already more than 15% away
                take_profit = entry_price - (1.5 * risk)  # Adjust take profit down
            else:
                stop_loss = entry_price + (reward / 1.5)  # Adjust stop loss up
        
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
        delta = option_features['delta']
        days_to_expiration = option_features['days_to_expiration']
        implied_volatility = option_features['implied_volatility']
        
        # Try to use ML prediction if available
        try:
            if hasattr(self, 'ml_integration') and self.ml_integration:
                ml_prediction = self.ml_integration.predict(features)
                if ml_prediction and 'exit_probability' in ml_prediction:
                    return ml_prediction['exit_probability']
        except Exception as e:
            self.logger.error(f"Error in ML prediction for exit probability: {str(e)}")
        
        # If no ML prediction, use enhanced rule-based approach
        
        # Base probability on delta (approximation of ITM probability)
        base_probability = abs(delta)
        
        # Adjust based on days to expiration
        if days_to_expiration < 7:
            # Short-dated options have less time for movement
            dte_factor = 0.8
        elif days_to_expiration > 45:
            # Long-dated options have more time for movement
            dte_factor = 1.2
        else:
            dte_factor = 1.0
        
        # Adjust based on implied volatility
        if implied_volatility > 0.5:
            # High volatility increases uncertainty
            iv_factor = 0.9
        elif implied_volatility < 0.2:
            # Low volatility increases certainty
            iv_factor = 1.1
        else:
            iv_factor = 1.0
        
        # Calculate adjusted probability
        adjusted_probability = base_probability * dte_factor * iv_factor
        
        # Ensure probability is between 0 and 1
        exit_probability = max(0.1, min(adjusted_probability, 0.95))
        
        return exit_probability
    
    def _calculate_profit_projections(self, entry_price, price_targets, option_data, position_type):
        """
        Calculate detailed profit projections for different exit scenarios.
        
        Args:
            entry_price (float): Entry price of the position
            price_targets (list): List of price targets
            option_data (dict): Option contract data
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            dict: Profit projections for different scenarios
        """
        self.logger.info("Calculating profit projections")
        
        # Get option features
        days_to_expiration = option_data.get('daysToExpiration', 30)
        
        # Initialize projections
        projections = {
            'scenarios': [],
            'max_profit_potential': 0,
            'max_loss_potential': 0,
            'expected_value': 0
        }
        
        # Calculate profit for each price target
        for i, target in enumerate(price_targets):
            target_price = target['price']
            percentage = target['percentage']
            
            # Calculate profit
            if position_type == 'long':
                profit = (target_price - entry_price) * percentage
                profit_percentage = ((target_price / entry_price) - 1) * 100
            else:
                profit = (entry_price - target_price) * percentage
                profit_percentage = ((entry_price / target_price) - 1) * 100
            
            # Calculate probability based on target level
            probability = 0.8 / (i + 1)  # Higher targets have lower probability
            
            # Calculate days to target based on target level
            days_to_target = min(days_to_expiration * (i + 1) / len(price_targets), days_to_expiration - 1)
            
            # Add scenario
            projections['scenarios'].append({
                'description': f"Target {i+1}",
                'exit_price': target_price,
                'profit': profit,
                'profit_percentage': profit_percentage,
                'probability': probability,
                'days_to_target': days_to_target,
                'annualized_return': self._calculate_annualized_return(profit_percentage, days_to_target)
            })
        
        # Add worst-case scenario (stop loss hit)
        stop_loss = option_data.get('stop_loss', entry_price * 0.8 if position_type == 'long' else entry_price * 1.2)
        if position_type == 'long':
            loss = (stop_loss - entry_price)
            loss_percentage = ((stop_loss / entry_price) - 1) * 100
        else:
            loss = (entry_price - stop_loss)
            loss_percentage = ((entry_price / stop_loss) - 1) * 100
        
        projections['scenarios'].append({
            'description': "Stop Loss",
            'exit_price': stop_loss,
            'profit': loss,  # This will be negative
            'profit_percentage': loss_percentage,  # This will be negative
            'probability': 0.2,  # 20% chance of hitting stop loss
            'days_to_target': days_to_expiration * 0.2,  # Assume early stop loss
            'annualized_return': self._calculate_annualized_return(loss_percentage, days_to_expiration * 0.2)
        })
        
        # Calculate max profit potential (best scenario)
        best_scenario = max(projections['scenarios'], key=lambda x: x['profit'])
        projections['max_profit_potential'] = best_scenario['profit']
        
        # Calculate max loss potential (worst scenario)
        worst_scenario = min(projections['scenarios'], key=lambda x: x['profit'])
        projections['max_loss_potential'] = worst_scenario['profit']
        
        # Calculate expected value (probability-weighted average)
        expected_value = sum(scenario['profit'] * scenario['probability'] for scenario in projections['scenarios'])
        projections['expected_value'] = expected_value
        
        # Calculate risk-reward ratio
        if projections['max_loss_potential'] < 0:  # Ensure loss is negative
            projections['risk_reward_ratio'] = abs(projections['max_profit_potential'] / projections['max_loss_potential'])
        else:
            projections['risk_reward_ratio'] = 1.0
        
        return projections
    
    def _calculate_annualized_return(self, percentage_return, days_held):
        """
        Calculate annualized return from percentage return and days held.
        
        Args:
            percentage_return (float): Percentage return
            days_held (float): Number of days position is held
            
        Returns:
            float: Annualized return percentage
        """
        if days_held <= 0:
            return 0
        
        # Convert percentage to decimal
        decimal_return = percentage_return / 100
        
        # Calculate annualized return
        annualized_return = ((1 + decimal_return) ** (365 / days_held)) - 1
        
        # Convert back to percentage
        return annualized_return * 100
    
    def update_exit_strategy_with_new_data(self, exit_strategy, current_price, days_held):
        """
        Update exit strategy with new market data.
        
        Args:
            exit_strategy (dict): Original exit strategy
            current_price (float): Current price of the option
            days_held (int): Number of days position has been held
            
        Returns:
            dict: Updated exit strategy
        """
        self.logger.info(f"Updating exit strategy for {exit_strategy.get('symbol', 'unknown')}")
        
        # Get original values
        entry_price = exit_strategy.get('entry_price', 0)
        original_days_to_hold = exit_strategy.get('days_to_hold', 7)
        position_type = exit_strategy.get('position_type', 'long')
        
        # Calculate current profit/loss
        if position_type == 'long':
            current_pnl_percentage = ((current_price / entry_price) - 1) * 100
        else:
            current_pnl_percentage = ((entry_price / current_price) - 1) * 100
        
        # Determine if exit recommendation should be updated
        update_needed = False
        update_reason = []
        
        # Check if we've reached a price target
        price_targets = exit_strategy.get('price_targets', [])
        for target in price_targets:
            target_price = target.get('price', 0)
            if (position_type == 'long' and current_price >= target_price) or \
               (position_type == 'short' and current_price <= target_price):
                update_needed = True
                update_reason.append(f"Price target {target_price:.2f} reached")
                break
        
        # Check if we've held longer than recommended
        if days_held >= original_days_to_hold:
            update_needed = True
            update_reason.append(f"Recommended holding period ({original_days_to_hold} days) reached")
        
        # Check if we're approaching expiration
        option_data = {
            'symbol': exit_strategy.get('symbol', ''),
            'underlying': exit_strategy.get('underlying', ''),
            'option_type': exit_strategy.get('option_type', ''),
            'strike': exit_strategy.get('strike', 0),
            'expiration_date': exit_strategy.get('expiration_date', ''),
            'daysToExpiration': exit_strategy.get('days_to_expiration', 30) - days_held,
            'implied_volatility': exit_strategy.get('implied_volatility', 0.3),
            'delta': exit_strategy.get('delta', 0.5),
            'gamma': exit_strategy.get('gamma', 0.05),
            'theta': exit_strategy.get('theta', -0.05),
            'vega': exit_strategy.get('vega', 0.1),
            'rho': exit_strategy.get('rho', 0.01)
        }
        
        # If update needed, generate new exit strategy
        if update_needed:
            # Create updated exit strategy
            updated_strategy = self.predict_exit_strategy(
                option_data, entry_price, None, position_type
            )
            
            # Add update information
            updated_strategy['update_reason'] = update_reason
            updated_strategy['previous_days_to_hold'] = original_days_to_hold
            updated_strategy['days_held_so_far'] = days_held
            updated_strategy['current_pnl_percentage'] = current_pnl_percentage
            
            return updated_strategy
        else:
            # No update needed, return original with current P&L
            exit_strategy['current_pnl_percentage'] = current_pnl_percentage
            exit_strategy['days_held_so_far'] = days_held
            exit_strategy['days_remaining'] = max(0, original_days_to_hold - days_held)
            
            return exit_strategy
