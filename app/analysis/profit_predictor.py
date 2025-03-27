"""
Options profit prediction module for options recommendation platform.
Implements profit projection with time decay modeling and probability calculations.
"""
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class ProfitPredictor:
    """
    Class to predict options profit potential with time decay modeling
    """
    
    def __init__(self, options_analysis=None):
        """
        Initialize the profit predictor
        
        Args:
            options_analysis: OptionsAnalysis instance for Greeks calculations
        """
        self.options_analysis = options_analysis
        self.risk_free_rate = 0.05  # Default risk-free rate (5%)
        self.simulation_paths = 1000  # Default number of Monte Carlo paths
    
    def project_option_price(self, option_data, days_forward, volatility_adjustment=1.0):
        """
        Project option price forward in time accounting for theta decay and volatility
        
        Args:
            option_data (dict): Option data including price, strike, expiration, etc.
            days_forward (int): Number of days to project forward
            volatility_adjustment (float): Adjustment factor for implied volatility
            
        Returns:
            dict: Projected option data including price, decay, etc.
        """
        # Extract current option data
        try:
            # Get required parameters
            S = option_data.get('underlyingPrice', 0)
            K = option_data.get('strikePrice', 0)
            
            # Handle different date formats
            if 'daysToExpiration' in option_data:
                # If days to expiration is already provided
                if isinstance(option_data['daysToExpiration'], (int, float)):
                    days_to_expiration = float(option_data['daysToExpiration'])
                elif isinstance(option_data['daysToExpiration'], pd.Timedelta):
                    days_to_expiration = option_data['daysToExpiration'].days
                else:
                    # Try to parse as string
                    try:
                        days_to_expiration = float(option_data['daysToExpiration'])
                    except (ValueError, TypeError):
                        days_to_expiration = 30  # Default fallback
            elif 'expirationDate' in option_data:
                # Calculate days to expiration from expiration date
                if isinstance(option_data['expirationDate'], datetime):
                    expiration_date = option_data['expirationDate']
                else:
                    # Try to parse as string
                    try:
                        expiration_date = pd.to_datetime(option_data['expirationDate'])
                    except:
                        # Fallback to 30 days from now
                        expiration_date = datetime.now() + timedelta(days=30)
                
                days_to_expiration = (expiration_date - datetime.now()).days
            else:
                # Default fallback
                days_to_expiration = 30
            
            # Ensure days_to_expiration is positive
            days_to_expiration = max(1, days_to_expiration)
            
            # Get option type
            option_type = option_data.get('optionType', 'CALL').lower()
            
            # Get implied volatility or use historical volatility as fallback
            if 'impliedVolatility' in option_data and option_data['impliedVolatility'] > 0:
                sigma = option_data['impliedVolatility'] * volatility_adjustment
            else:
                # Default to moderate volatility if not available
                sigma = 0.3 * volatility_adjustment
            
            # Get current price or calculate using Black-Scholes
            if 'entryPrice' in option_data and option_data['entryPrice'] > 0:
                current_price = option_data['entryPrice']
            elif 'bid' in option_data and 'ask' in option_data:
                current_price = (option_data['bid'] + option_data['ask']) / 2
            else:
                # Calculate using Black-Scholes if we have the options_analysis module
                if self.options_analysis:
                    current_price = self.options_analysis.calculate_black_scholes(
                        S, K, days_to_expiration / 365, self.risk_free_rate, sigma, option_type
                    )
                else:
                    # Simple approximation if options_analysis not available
                    if option_type == 'call':
                        current_price = max(0, S - K) + (S * 0.05)  # Intrinsic + time value approximation
                    else:
                        current_price = max(0, K - S) + (S * 0.05)  # Intrinsic + time value approximation
            
            # Current time to expiration in years
            T_current = days_to_expiration / 365
            
            # Future time to expiration in years
            T_future = (days_to_expiration - days_forward) / 365
            
            # Ensure T_future is positive
            T_future = max(0.001, T_future)
            
            # Calculate projected price using Black-Scholes
            if self.options_analysis:
                future_price = self.options_analysis.calculate_black_scholes(
                    S, K, T_future, self.risk_free_rate, sigma, option_type
                )
            else:
                # Simple approximation if options_analysis not available
                # Time value decays approximately with square root of time
                time_value_decay = math.sqrt(T_future / T_current) if T_current > 0 else 0
                intrinsic_value = max(0, S - K) if option_type == 'call' else max(0, K - S)
                time_value = current_price - intrinsic_value
                future_price = intrinsic_value + (time_value * time_value_decay)
            
            # Calculate expected theta decay
            expected_decay = current_price - future_price
            
            # Calculate decay percentage
            decay_percentage = (expected_decay / current_price) * 100 if current_price > 0 else 0
            
            # Calculate daily theta (average daily decay)
            daily_theta = expected_decay / days_forward if days_forward > 0 else 0
            
            return {
                'symbol': option_data.get('symbol', ''),
                'optionType': option_data.get('optionType', ''),
                'strikePrice': K,
                'underlyingPrice': S,
                'daysToExpiration': days_to_expiration,
                'impliedVolatility': sigma,
                'current_price': current_price,
                'projected_price': future_price,
                'expected_decay': expected_decay,
                'decay_percentage': decay_percentage,
                'daily_theta': daily_theta,
                'days_forward': days_forward
            }
            
        except Exception as e:
            print(f"Error projecting option price: {str(e)}")
            return {
                'symbol': option_data.get('symbol', ''),
                'optionType': option_data.get('optionType', ''),
                'strikePrice': option_data.get('strikePrice', 0),
                'error': str(e)
            }
    
    def calculate_profit_probability(self, option_data, target_profit_pct=50, days_forward=None, num_simulations=None):
        """
        Calculate probability of achieving target profit using Monte Carlo simulation
        
        Args:
            option_data (dict): Option data including price, strike, expiration, etc.
            target_profit_pct (float): Target profit percentage
            days_forward (int): Number of days to project forward (default: half of days to expiration)
            num_simulations (int): Number of Monte Carlo simulations (default: self.simulation_paths)
            
        Returns:
            dict: Probability analysis including profit probability, expected return, etc.
        """
        try:
            # Extract required parameters
            S = option_data.get('underlyingPrice', 0)
            K = option_data.get('strikePrice', 0)
            
            # Get days to expiration
            if 'daysToExpiration' in option_data:
                if isinstance(option_data['daysToExpiration'], (int, float)):
                    days_to_expiration = float(option_data['daysToExpiration'])
                elif isinstance(option_data['daysToExpiration'], pd.Timedelta):
                    days_to_expiration = option_data['daysToExpiration'].days
                else:
                    try:
                        days_to_expiration = float(option_data['daysToExpiration'])
                    except (ValueError, TypeError):
                        days_to_expiration = 30
            elif 'expirationDate' in option_data:
                if isinstance(option_data['expirationDate'], datetime):
                    expiration_date = option_data['expirationDate']
                else:
                    try:
                        expiration_date = pd.to_datetime(option_data['expirationDate'])
                    except:
                        expiration_date = datetime.now() + timedelta(days=30)
                
                days_to_expiration = (expiration_date - datetime.now()).days
            else:
                days_to_expiration = 30
            
            # Ensure days_to_expiration is positive
            days_to_expiration = max(1, days_to_expiration)
            
            # Set days_forward if not provided (default to half of days to expiration)
            if days_forward is None:
                days_forward = max(1, days_to_expiration // 2)
            
            # Set number of simulations if not provided
            if num_simulations is None:
                num_simulations = self.simulation_paths
            
            # Get option type
            option_type = option_data.get('optionType', 'CALL').lower()
            
            # Get implied volatility or use historical volatility as fallback
            if 'impliedVolatility' in option_data and option_data['impliedVolatility'] > 0:
                sigma = option_data['impliedVolatility']
            else:
                sigma = 0.3  # Default to moderate volatility
            
            # Get current price
            if 'entryPrice' in option_data and option_data['entryPrice'] > 0:
                current_price = option_data['entryPrice']
            elif 'bid' in option_data and 'ask' in option_data:
                current_price = (option_data['bid'] + option_data['ask']) / 2
            else:
                if self.options_analysis:
                    current_price = self.options_analysis.calculate_black_scholes(
                        S, K, days_to_expiration / 365, self.risk_free_rate, sigma, option_type
                    )
                else:
                    if option_type == 'call':
                        current_price = max(0, S - K) + (S * 0.05)
                    else:
                        current_price = max(0, K - S) + (S * 0.05)
            
            # Calculate target price
            target_price = current_price * (1 + target_profit_pct / 100)
            
            # Time parameters
            T = days_forward / 365.0  # Time in years
            dt = T / 252  # Daily time step (252 trading days per year)
            
            # Monte Carlo simulation
            num_steps = int(days_forward)
            price_paths = np.zeros((num_simulations, num_steps + 1))
            option_price_paths = np.zeros((num_simulations, num_steps + 1))
            
            # Initialize with current prices
            price_paths[:, 0] = S
            option_price_paths[:, 0] = current_price
            
            # Generate random price paths
            for i in range(num_simulations):
                for j in range(1, num_steps + 1):
                    # Generate random return
                    z = np.random.normal(0, 1)
                    
                    # Update stock price using geometric Brownian motion
                    price_paths[i, j] = price_paths[i, j-1] * np.exp((self.risk_free_rate - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
                    
                    # Calculate option price at this step
                    days_remaining = days_to_expiration - j
                    if days_remaining <= 0:
                        # At expiration, option is worth intrinsic value
                        if option_type == 'call':
                            option_price_paths[i, j] = max(0, price_paths[i, j] - K)
                        else:
                            option_price_paths[i, j] = max(0, K - price_paths[i, j])
                    else:
                        # Before expiration, calculate using Black-Scholes
                        if self.options_analysis:
                            option_price_paths[i, j] = self.options_analysis.calculate_black_scholes(
                                price_paths[i, j], K, days_remaining / 365, self.risk_free_rate, sigma, option_type
                            )
                        else:
                            # Simple approximation
                            time_decay_factor = math.sqrt(days_remaining / days_to_expiration)
                            intrinsic_value = max(0, price_paths[i, j] - K) if option_type == 'call' else max(0, K - price_paths[i, j])
                            time_value = (current_price - max(0, S - K) if option_type == 'call' else current_price - max(0, K - S)) * time_decay_factor
                            option_price_paths[i, j] = intrinsic_value + time_value
            
            # Calculate maximum price reached for each path
            max_option_prices = np.max(option_price_paths, axis=1)
            
            # Calculate probability of reaching target price
            paths_reaching_target = np.sum(max_option_prices >= target_price)
            probability_of_target = paths_reaching_target / num_simulations
            
            # Calculate expected return
            final_option_prices = option_price_paths[:, -1]
            expected_return = np.mean(final_option_prices / current_price - 1) * 100
            
            # Calculate risk metrics
            std_dev_return = np.std(final_option_prices / current_price - 1) * 100
            sharpe_ratio = expected_return / std_dev_return if std_dev_return > 0 else 0
            
            # Calculate win rate (percentage of paths that end with profit)
            win_paths = np.sum(final_option_prices > current_price)
            win_rate = win_paths / num_simulations
            
            # Calculate average profit and loss
            profit_paths = final_option_prices[final_option_prices > current_price]
            loss_paths = final_option_prices[final_option_prices <= current_price]
            
            avg_profit_pct = np.mean((profit_paths / current_price - 1) * 100) if len(profit_paths) > 0 else 0
            avg_loss_pct = np.mean((loss_paths / current_price - 1) * 100) if len(loss_paths) > 0 else 0
            
            # Calculate profit factor (average profit / average loss)
            profit_factor = abs(avg_profit_pct / avg_loss_pct) if avg_loss_pct < 0 else float('inf')
            
            return {
                'symbol': option_data.get('symbol', ''),
                'optionType': option_data.get('optionType', ''),
                'strikePrice': K,
                'underlyingPrice': S,
                'current_price': current_price,
                'target_price': target_price,
                'target_profit_pct': target_profit_pct,
                'days_forward': days_forward,
                'probability_of_target': probability_of_target,
                'expected_return_pct': expected_return,
                'win_rate': win_rate,
                'avg_profit_pct': avg_profit_pct,
                'avg_loss_pct': avg_loss_pct,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'std_dev_return': std_dev_return
            }
            
        except Exception as e:
            print(f"Error calculating profit probability: {str(e)}")
            return {
                'symbol': option_data.get('symbol', ''),
                'optionType': option_data.get('optionType', ''),
                'strikePrice': option_data.get('strikePrice', 0),
                'error': str(e)
            }
    
    def find_optimal_exit_points(self, option_data, risk_tolerance='moderate'):
        """
        Find optimal exit points (take profit and stop loss) based on risk profile
        
        Args:
            option_data (dict): Option data including price, strike, expiration, etc.
            risk_tolerance (str): Risk tolerance level ('conservative', 'moderate', 'aggressive')
            
        Returns:
            dict: Optimal exit points including take profit and stop loss levels
        """
        try:
            # Get current price
            if 'entryPrice' in option_data and option_data['entryPrice'] > 0:
                current_price = option_data['entryPrice']
            elif 'bid' in option_data and 'ask' in option_data:
                current_price = (option_data['bid'] + option_data['ask']) / 2
            else:
                current_price = 1.0  # Default fallback
            
            # Set risk parameters based on risk tolerance
            if risk_tolerance == 'conservative':
                take_profit_pct = 25
                stop_loss_pct = -15
                time_decay_threshold_pct = 10
            elif risk_tolerance == 'aggressive':
                take_profit_pct = 100
                stop_loss_pct = -40
                time_decay_threshold_pct = 25
            else:  # moderate (default)
                take_profit_pct = 50
                stop_loss_pct = -25
                time_decay_threshold_pct = 15
            
            # Calculate exit prices
            take_profit_price = current_price * (1 + take_profit_pct / 100)
            stop_loss_price = current_price * (1 + stop_loss_pct / 100)
            
            # Get days to expiration
            if 'daysToExpiration' in option_data:
                if isinstance(option_data['daysToExpiration'], (int, float)):
                    days_to_expiration = float(option_data['daysToExpiration'])
                elif isinstance(option_data['daysToExpiration'], pd.Timedelta):
                    days_to_expiration = option_data['daysToExpiration'].days
                else:
                    try:
                        days_to_expiration = float(option_data['daysToExpiration'])
                    except (ValueError, TypeError):
                        days_to_expiration = 30
            elif 'expirationDate' in option_data:
                if isinstance(option_data['expirationDate'], datetime):
                    expiration_date = option_data['expirationDate']
                else:
                    try:
                        expiration_date = pd.to_datetime(option_data['expirationDate'])
                    except:
                        expiration_date = datetime.now() + timedelta(days=30)
                
                days_to_expiration = (expiration_date - datetime.now()).days
            else:
                days_to_expiration = 30
            
            # Calculate time-based exit (days before expiration to exit)
            time_exit_days = max(1, int(days_to_expiration * 0.2))  # Exit at 20% of time remaining
            
            # Calculate time decay threshold
            time_decay_threshold = current_price * (time_decay_threshold_pct / 100)
            
            # Calculate optimal days to hold based on theta decay
            if 'theta' in option_data and option_data['theta'] < 0:
                # If theta is available, use it to calculate optimal hold time
                daily_theta = option_data['theta']
                days_to_threshold = time_decay_threshold / abs(daily_theta)
                optimal_hold_days = min(days_to_expiration - time_exit_days, int(days_to_threshold))
            else:
                # Default to 1/3 of days to expiration
                optimal_hold_days = max(1, int(days_to_expiration / 3))
            
            return {
                'symbol': option_data.get('symbol', ''),
                'optionType': option_data.get('optionType', ''),
                'strikePrice': option_data.get('strikePrice', 0),
                'current_price': current_price,
                'take_profit_price': take_profit_price,
                'take_profit_pct': take_profit_pct,
                'stop_loss_price': stop_loss_price,
                'stop_loss_pct': stop_loss_pct,
                'time_exit_days': time_exit_days,
                'optimal_hold_days': optimal_hold_days,
                'time_decay_threshold': time_decay_threshold,
                'risk_tolerance': risk_tolerance
            }
            
        except Exception as e:
            print(f"Error finding optimal exit points: {str(e)}")
            return {
                'symbol': option_data.get('symbol', ''),
                'optionType': option_data.get('optionType', ''),
                'strikePrice': option_data.get('strikePrice', 0),
                'error': str(e)
            }
    
    def analyze_option_profit_potential(self, option_data, risk_tolerance='moderate'):
        """
        Comprehensive analysis of option profit potential
        
        Args:
            option_data (dict): Option data including price, strike, expiration, etc.
            risk_tolerance (str): Risk tolerance level ('conservative', 'moderate', 'aggressive')
            
        Returns:
            dict: Comprehensive profit analysis
        """
        try:
            # Get days to expiration
            if 'daysToExpiration' in option_data:
                if isinstance(option_data['daysToExpiration'], (int, float)):
                    days_to_expiration = float(option_data['daysToExpiration'])
                elif isinstance(option_data['daysToExpiration'], pd.Timedelta):
                    days_to_expiration = option_data['daysToExpiration'].days
                else:
                    try:
                        days_to_expiration = float(option_data['daysToExpiration'])
                    except (ValueError, TypeError):
                        days_to_expiration = 30
            elif 'expirationDate' in option_data:
                if isinstance(option_data['expirationDate'], datetime):
                    expiration_date = option_data['expirationDate']
                else:
                    try:
                        expiration_date = pd.to_datetime(option_data['expirationDate'])
                    except:
                        expiration_date = datetime.now() + timedelta(days=30)
                
                days_to_expiration = (expiration_date - datetime.now()).days
            else:
                days_to_expiration = 30
            
            # Set days forward based on days to expiration
            if days_to_expiration <= 7:
                days_forward = 1  # For very short-term options
            elif days_to_expiration <= 30:
                days_forward = 7  # For short-term options
            elif days_to_expiration <= 90:
                days_forward = 14  # For medium-term options
            else:
                days_forward = 30  # For long-term options
            
            # Set target profit based on risk tolerance
            if risk_tolerance == 'conservative':
                target_profit_pct = 25
            elif risk_tolerance == 'aggressive':
                target_profit_pct = 100
            else:  # moderate (default)
                target_profit_pct = 50
            
            # Project option price
            price_projection = self.project_option_price(option_data, days_forward)
            
            # Calculate profit probability
            profit_probability = self.calculate_profit_probability(
                option_data, target_profit_pct=target_profit_pct, days_forward=days_forward
            )
            
            # Find optimal exit points
            exit_points = self.find_optimal_exit_points(option_data, risk_tolerance)
            
            # Combine results
            analysis = {
                'symbol': option_data.get('symbol', ''),
                'optionType': option_data.get('optionType', ''),
                'strikePrice': option_data.get('strikePrice', 0),
                'underlyingPrice': option_data.get('underlyingPrice', 0),
                'current_price': price_projection.get('current_price', 0),
                'days_to_expiration': days_to_expiration,
                'price_projection': {
                    'days_forward': days_forward,
                    'projected_price': price_projection.get('projected_price', 0),
                    'expected_decay': price_projection.get('expected_decay', 0),
                    'decay_percentage': price_projection.get('decay_percentage', 0),
                    'daily_theta': price_projection.get('daily_theta', 0)
                },
                'profit_probability': {
                    'target_profit_pct': target_profit_pct,
                    'probability_of_target': profit_probability.get('probability_of_target', 0),
                    'expected_return_pct': profit_probability.get('expected_return_pct', 0),
                    'win_rate': profit_probability.get('win_rate', 0),
                    'profit_factor': profit_probability.get('profit_factor', 0),
                    'sharpe_ratio': profit_probability.get('sharpe_ratio', 0)
                },
                'exit_strategy': {
                    'take_profit_price': exit_points.get('take_profit_price', 0),
                    'take_profit_pct': exit_points.get('take_profit_pct', 0),
                    'stop_loss_price': exit_points.get('stop_loss_price', 0),
                    'stop_loss_pct': exit_points.get('stop_loss_pct', 0),
                    'optimal_hold_days': exit_points.get('optimal_hold_days', 0),
                    'time_exit_days': exit_points.get('time_exit_days', 0)
                },
                'risk_tolerance': risk_tolerance
            }
            
            # Calculate overall profit score (0-100)
            profit_score = (
                profit_probability.get('probability_of_target', 0) * 40 +  # 40% weight
                min(1, profit_probability.get('expected_return_pct', 0) / 100) * 30 +  # 30% weight
                profit_probability.get('win_rate', 0) * 20 +  # 20% weight
                min(1, profit_probability.get('sharpe_ratio', 0)) * 10  # 10% weight
            ) * 100
            
            analysis['profit_score'] = profit_score
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing option profit potential: {str(e)}")
            return {
                'symbol': option_data.get('symbol', ''),
                'optionType': option_data.get('optionType', ''),
                'strikePrice': option_data.get('strikePrice', 0),
                'error': str(e)
            }
