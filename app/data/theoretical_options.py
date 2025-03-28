"""
Theoretical Options Data Module

This module provides functionality to generate theoretical options data based on underlying asset prices.
It uses the Black-Scholes model to calculate theoretical option prices and Greeks.
It prioritizes using Schwab API data for underlying assets.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('theoretical_options')

class TheoreticalOptionsGenerator:
    def __init__(self, db):
        self.db = db
        logger.info("Initialized theoretical options generator")
        
    def generate_theoretical_history(self, symbol, option_details, start_date=None, end_date=None, current_iv=None):
        """Generate theoretical options history based on underlying price history
        
        Args:
            symbol (str): Underlying symbol (e.g., 'SPY')
            option_details (dict): Details of the option contract including:
                - strike_price: Strike price of the option
                - option_type: 'CALL' or 'PUT'
                - expiration_date: Expiration date of the option
            start_date (str, optional): Start date for historical data
            end_date (str, optional): End date for historical data
            current_iv (float, optional): Current implied volatility to use
            
        Returns:
            pd.DataFrame: Theoretical options data
        """
        logger.info(f"Generating theoretical history for {symbol} option using Schwab API data")
        
        # Get underlying price history from Schwab API via database
        underlying_history = self.db.get_historical_underlying(symbol, start_date=start_date, end_date=end_date)
        
        if underlying_history.empty:
            logger.warning(f"No historical data available for underlying {symbol} from Schwab API")
            return pd.DataFrame()
            
        logger.info(f"Found {len(underlying_history)} historical price points for {symbol} from Schwab API")
        
        # Extract option details
        strike = option_details.get('strike_price')
        option_type = option_details.get('option_type', 'CALL').upper()
        expiration_date = option_details.get('expiration_date')
        option_symbol = option_details.get('symbol', f"{symbol}_{expiration_date.replace('-', '')}{'C' if option_type == 'CALL' else 'P'}{int(strike)}")
        
        if not all([strike, option_type, expiration_date]):
            logger.error("Missing required option details")
            return pd.DataFrame()
        
        # If no current IV provided, estimate from historical volatility
        if current_iv is None:
            # Calculate historical volatility (20-day rolling)
            returns = underlying_history['close'].pct_change().dropna()
            hist_vol = returns.rolling(20).std() * np.sqrt(252)
            current_iv = hist_vol.iloc[-1] if not hist_vol.empty else 0.2
            logger.info(f"Using estimated IV of {current_iv:.2f} based on historical volatility")
        
        # Risk-free rate (simplified)
        risk_free_rate = 0.03
        
        # Generate theoretical options data for each day
        theoretical_data = []
        
        for _, row in underlying_history.iterrows():
            date = row['timestamp']
            price = row['close']
            
            # Calculate days to expiry
            try:
                current_date = datetime.fromisoformat(date.split('T')[0] if 'T' in date else date)
                expiry_date = datetime.fromisoformat(expiration_date.split('T')[0] if 'T' in expiration_date else expiration_date)
                days_to_expiry = (expiry_date - current_date).days
                
                # Skip if already expired
                if days_to_expiry < 0:
                    continue
                    
                # Time to expiry in years
                t = days_to_expiry / 365
            except Exception as e:
                logger.error(f"Error calculating days to expiry: {e}")
                continue
            
            # Calculate option price using Black-Scholes
            try:
                # Calculate option price
                option_price = self._black_scholes(price, strike, t, risk_free_rate, current_iv, option_type.lower())
                
                # Calculate Greeks
                delta, gamma, theta, vega = self._calculate_greeks(price, strike, t, risk_free_rate, current_iv, option_type.lower())
                
                # Add to theoretical data
                theoretical_data.append({
                    'symbol': option_symbol,
                    'underlying': symbol,
                    'strike': strike,
                    'option_type': option_type,
                    'expiration_date': expiration_date,
                    'bid': option_price * 0.98,  # Simulate bid-ask spread
                    'ask': option_price * 1.02,
                    'last': option_price,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'implied_volatility': current_iv,
                    'timestamp': date,
                    'days_to_expiry': days_to_expiry,
                    'underlying_price': price,
                    'is_theoretical': True  # Flag to indicate this is theoretical data
                })
            except Exception as e:
                logger.error(f"Error calculating option price: {e}")
                continue
        
        logger.info(f"Generated {len(theoretical_data)} theoretical options records using Schwab API data")
        return pd.DataFrame(theoretical_data)
    
    def _black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option price using Black-Scholes formula
        
        Args:
            S (float): Underlying price
            K (float): Strike price
            T (float): Time to expiry in years
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Option price
        """
        try:
            if T <= 0:
                # For expired options, return intrinsic value
                if option_type.lower() == 'call':
                    return max(0, S - K)
                else:
                    return max(0, K - S)
                    
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
            return price
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")
            return 0
    
    def _calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks using Black-Scholes formula
        
        Args:
            S (float): Underlying price
            K (float): Strike price
            T (float): Time to expiry in years
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            
        Returns:
            tuple: (delta, gamma, theta, vega)
        """
        try:
            if T <= 0:
                # For expired options, return zeros
                return 0, 0, 0, 0
                
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
                
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type.lower() == 'call':
                theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            else:
                theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            
            # Vega (same for calls and puts)
            vega = S * np.sqrt(T) * norm.pdf(d1)
            
            return delta, gamma, theta, vega
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return 0, 0, 0, 0
