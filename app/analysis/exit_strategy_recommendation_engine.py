"""
Enhanced Recommendation Engine Integration for Exit Strategy Prediction

This module integrates the exit strategy predictor with the enhanced recommendation engine
to provide complete entry and exit recommendations for options trading.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import original recommendation engine
from app.analysis.enhanced_recommendation_engine import EnhancedRecommendationEngine

# Import exit strategy predictors
from app.analysis.exit_strategy_predictor import ExitStrategyPredictor
from app.analysis.exit_timing_predictor import ExitTimingPredictor
from app.analysis.price_target_predictor import PriceTargetPredictor

# Import utils
from app.utils.enhanced_logging import EnhancedLogger

class ExitStrategyEnhancedRecommendationEngine(EnhancedRecommendationEngine):
    """
    Enhanced recommendation engine with integrated exit strategy prediction.
    
    This class extends the EnhancedRecommendationEngine to include exit strategy
    predictions, providing recommendations for both when to buy options and
    when/at what premium to sell them.
    """
    
    def __init__(self, data_collector, ml_config_path=None, debug=False):
        """
        Initialize the exit strategy enhanced recommendation engine.
        
        Args:
            data_collector: The data collector instance
            ml_config_path (str, optional): Path to ML configuration file
            debug (bool, optional): Whether to enable debug output
        """
        # Initialize the parent class
        super().__init__(data_collector, ml_config_path, debug)
        
        # Initialize logger
        self.logger = EnhancedLogger('exit_strategy_recommendation_engine')
        if debug:
            self.logger.logger.setLevel('DEBUG')
        
        # Initialize exit strategy predictor
        self.exit_strategy_predictor = ExitStrategyPredictor(
            ml_integration=self.ml_integration,
            data_collector=data_collector,
            config_path=ml_config_path
        )
        
        # Initialize specialized predictors
        self.exit_timing_predictor = ExitTimingPredictor(
            data_collector=data_collector,
            config_path=ml_config_path
        )
        
        self.price_target_predictor = PriceTargetPredictor(
            data_collector=data_collector,
            config_path=ml_config_path
        )
        
        self.logger.info("Exit Strategy Enhanced Recommendation Engine initialized")
    
    def generate_recommendations(self, symbol=None, lookback_days=30, confidence_threshold=0.6, strategy_types=None, symbols=None, strategy_type='all', max_recommendations=10):
        """
        Generate enhanced options trading recommendations with exit strategies.
        
        Args:
            symbol (str, optional): Stock symbol to generate recommendations for (single symbol)
            lookback_days (int, optional): Number of days to look back for historical data
            confidence_threshold (float, optional): Minimum confidence threshold for recommendations
            strategy_types (list, optional): List of strategy types to consider
            symbols (list, optional): List of symbols to generate recommendations for (multiple symbols)
            strategy_type (str, optional): Type of strategy to recommend
            max_recommendations (int, optional): Maximum number of recommendations to return
            
        Returns:
            pandas.DataFrame: Enhanced recommendations with exit strategies
        """
        self.logger.info("Generating recommendations with exit strategies")
        
        # Get base recommendations from parent class
        recommendations = super().generate_recommendations(
            symbol, lookback_days, confidence_threshold, strategy_types, symbols, strategy_type, max_recommendations
        )
        
        # If no recommendations, return empty DataFrame
        if recommendations.empty:
            self.logger.info("No recommendations to enhance with exit strategies")
            return recommendations
        
        # Convert to list of dictionaries for easier processing
        rec_list = recommendations.to_dict('records')
        
        # Enhance recommendations with exit strategies
        enhanced_recs = []
        for rec in rec_list:
            try:
                # Extract option data
                option_data = self._extract_option_data(rec)
                
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
                exit_strategy = self.exit_strategy_predictor.predict_exit_strategy(
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
                
                enhanced_recs.append(enhanced_rec)
                
            except Exception as e:
                self.logger.error(f"Error enhancing recommendation with exit strategy: {str(e)}")
                enhanced_recs.append(rec)
        
        # Convert back to DataFrame
        enhanced_recommendations = pd.DataFrame(enhanced_recs)
        
        self.logger.info(f"Generated {len(enhanced_recommendations)} recommendations with exit strategies")
        return enhanced_recommendations
    
    def _extract_option_data(self, recommendation):
        """
        Extract option data from recommendation.
        
        Args:
            recommendation (dict): Recommendation dictionary
            
        Returns:
            dict: Option data dictionary
        """
        # Extract option data
        option_data = {
            'symbol': recommendation.get('symbol', ''),
            'underlying': recommendation.get('underlying', ''),
            'option_type': recommendation.get('option_type', 'CALL'),
            'strike': recommendation.get('strike', 0),
            'expiration_date': recommendation.get('expiration_date', ''),
            'bid': recommendation.get('bid', 0),
            'ask': recommendation.get('ask', 0),
            'last': recommendation.get('last', 0),
            'volume': recommendation.get('volume', 0),
            'open_interest': recommendation.get('open_interest', 0),
            'delta': recommendation.get('delta', 0.5),
            'gamma': recommendation.get('gamma', 0.05),
            'theta': recommendation.get('theta', -0.05),
            'vega': recommendation.get('vega', 0.1),
            'rho': recommendation.get('rho', 0.01),
            'implied_volatility': recommendation.get('implied_volatility', 0.3),
            'underlyingPrice': recommendation.get('underlyingPrice', 0)
        }
        
        return option_data
    
    def generate_exit_strategy_for_position(self, position_data):
        """
        Generate exit strategy for an existing options position.
        
        Args:
            position_data (dict): Position data including:
                - symbol: Option symbol
                - entry_price: Entry price of the position
                - entry_date: Entry date of the position
                - position_type: Type of position ('long' or 'short')
                - option_data: Option contract data
                
        Returns:
            dict: Exit strategy recommendation
        """
        self.logger.info(f"Generating exit strategy for position {position_data.get('symbol', 'unknown')}")
        
        try:
            # Extract required data
            symbol = position_data.get('symbol', '')
            entry_price = position_data.get('entry_price', 0)
            entry_date = position_data.get('entry_date', datetime.now())
            position_type = position_data.get('position_type', 'long')
            option_data = position_data.get('option_data', {})
            
            # If entry_date is a string, convert to datetime
            if isinstance(entry_date, str):
                try:
                    entry_date = datetime.strptime(entry_date, '%Y-%m-%d')
                except ValueError:
                    # Try alternative date formats
                    try:
                        entry_date = datetime.strptime(entry_date, '%Y-%m-%d %H:%M:%S')
                    except:
                        entry_date = datetime.now()
            
            # Generate exit strategy
            exit_strategy = self.exit_strategy_predictor.predict_exit_strategy(
                option_data, entry_price, entry_date, position_type
            )
            
            return exit_strategy
            
        except Exception as e:
            self.logger.error(f"Error generating exit strategy for position: {str(e)}")
            
            # Return a basic exit strategy as fallback
            return {
                'symbol': symbol,
                'optimal_exit_time': (datetime.now() + timedelta(days=7)).isoformat(),
                'days_to_hold': 7,
                'price_targets': [
                    {'price': entry_price * 1.2, 'percentage': 0.5, 'profit_percentage': 20},
                    {'price': entry_price * 1.5, 'percentage': 0.5, 'profit_percentage': 50}
                ],
                'stop_loss': entry_price * 0.8,
                'take_profit': entry_price * 1.5,
                'exit_probability': 0.5,
                'exit_reasons': ['Fallback exit strategy due to error'],
                'confidence_score': 0.3
            }
    
    def update_exit_strategy_with_new_data(self, position_data, current_price, days_held):
        """
        Update exit strategy with new market data.
        
        Args:
            position_data (dict): Position data
            current_price (float): Current price of the option
            days_held (int): Number of days position has been held
            
        Returns:
            dict: Updated exit strategy
        """
        self.logger.info(f"Updating exit strategy for position {position_data.get('symbol', 'unknown')}")
        
        try:
            # Extract required data
            symbol = position_data.get('symbol', '')
            entry_price = position_data.get('entry_price', 0)
            entry_date = position_data.get('entry_date', datetime.now() - timedelta(days=days_held))
            position_type = position_data.get('position_type', 'long')
            option_data = position_data.get('option_data', {})
            
            # If entry_date is a string, convert to datetime
            if isinstance(entry_date, str):
                try:
                    entry_date = datetime.strptime(entry_date, '%Y-%m-%d')
                except ValueError:
                    # Try alternative date formats
                    try:
                        entry_date = datetime.strptime(entry_date, '%Y-%m-%d %H:%M:%S')
                    except:
                        entry_date = datetime.now() - timedelta(days=days_held)
            
            # Calculate profit/loss so far
            if position_type == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Generate updated exit strategy
            exit_strategy = self.exit_strategy_predictor.predict_exit_strategy(
                option_data, entry_price, entry_date, position_type
            )
            
            # Adjust based on current P&L
            if pnl_pct > 0.5:  # If already up more than 50%
                # Consider taking profits sooner
                exit_strategy['days_to_hold'] = max(1, exit_strategy['days_to_hold'] // 2)
                exit_strategy['exit_reasons'].append(f"Adjusted for current profit of {pnl_pct:.1%}")
            elif pnl_pct < -0.2:  # If down more than 20%
                # Consider cutting losses
                exit_strategy['days_to_hold'] = max(1, exit_strategy['days_to_hold'] // 3)
                exit_strategy['exit_reasons'].append(f"Adjusted for current loss of {pnl_pct:.1%}")
            
            # Update optimal exit time based on adjusted days to hold
            exit_strategy['optimal_exit_time'] = (entry_date + timedelta(days=exit_strategy['days_to_hold'])).isoformat()
            
            return exit_strategy
            
        except Exception as e:
            self.logger.error(f"Error updating exit strategy: {str(e)}")
            
            # Return a basic exit strategy as fallback
            return {
                'symbol': symbol,
                'optimal_exit_time': (datetime.now() + timedelta(days=3)).isoformat(),
                'days_to_hold': 3,
                'price_targets': [
                    {'price': current_price * 1.1, 'percentage': 0.5, 'profit_percentage': 10},
                    {'price': current_price * 1.2, 'percentage': 0.5, 'profit_percentage': 20}
                ],
                'stop_loss': current_price * 0.9,
                'take_profit': current_price * 1.2,
                'exit_probability': 0.5,
                'exit_reasons': ['Fallback exit strategy due to error'],
                'confidence_score': 0.3
            }
