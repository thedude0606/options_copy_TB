"""
Synthetic Options Data Generator Module

This module provides a synthetic options data generator based on Black-Scholes model.
It generates synthetic options data when real historical data is not available.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('synthetic_data')

class SyntheticOptionsGenerator:
    def __init__(self, db):
        self.db = db
        logger.info("Initialized synthetic options generator")
        
    def generate_synthetic_history(self, symbol, start_date=None, end_date=None, strikes=None, expirations=None):
        """Generate synthetic options history based on underlying price history"""
        logger.info(f"Generating synthetic history for {symbol}")
        
        # Get underlying price history
        underlying_history = self.db.get_historical_underlying(symbol, start_date=start_date, end_date=end_date)
        
        if underlying_history.empty:
            logger.warning(f"No historical data available for {symbol}")
            return pd.DataFrame()
            
        logger.info(f"Found {len(underlying_history)} historical price points for {symbol}")
        
        # If strikes not provided, generate a range around the average price
        if strikes is None:
            avg_price = underlying_history['close'].mean()
            strikes = [avg_price * (1 + i/10) for i in range(-5, 6)]
            
        # If expirations not provided, generate standard expirations
        if expirations is None:
            expirations = [30, 60, 90, 180]
            
        logger.info(f"Using {len(strikes)} strikes and {len(expirations)} expirations")
        
        # Generate synthetic options data for each day
        synthetic_data = []
        
        for _, row in underlying_history.iterrows():
            date = row['timestamp']
            price = row['close']
            
            # Estimate historical volatility (20-day rolling)
            volatility = underlying_history['close'].pct_change().rolling(20).std() * np.sqrt(252)
            vol = volatility.loc[date] if date in volatility.index else 0.2  # Default if not enough history
            
            # Risk-free rate (simplified)
            risk_free_rate = 0.03
            
            # Generate options for each strike and expiration
            for strike in strikes:
                for days_to_expiry in expirations:
                    expiry_date = (datetime.fromisoformat(date) + timedelta(days=days_to_expiry)).isoformat()
                    
                    # Calculate call price using Black-Scholes
                    call_price = self._black_scholes(price, strike, days_to_expiry/365, risk_free_rate, vol, option_type='call')
                    
                    # Calculate put price using Black-Scholes
                    put_price = self._black_scholes(price, strike, days_to_expiry/365, risk_free_rate, vol, option_type='put')
                    
                    # Calculate Greeks
                    call_delta, call_gamma, call_theta, call_vega = self._calculate_greeks(price, strike, days_to_expiry/365, risk_free_rate, vol, option_type='call')
                    put_delta, put_gamma, put_theta, put_vega = self._calculate_greeks(price, strike, days_to_expiry/365, risk_free_rate, vol, option_type='put')
                    
                    # Add call option
                    synthetic_data.append({
                        'symbol': f"{symbol}_{expiry_date.replace('-', '')}C{int(strike)}",
                        'underlying': symbol,
                        'strike': strike,
                        'option_type': 'CALL',
                        'expiration_date': expiry_date,
                        'bid': call_price * 0.98,  # Simulate bid-ask spread
                        'ask': call_price * 1.02,
                        'last': call_price,
                        'delta': call_delta,
                        'gamma': call_gamma,
                        'theta': call_theta,
                        'vega': call_vega,
                        'implied_volatility': vol,
                        'timestamp': date,
                        'days_to_expiry': days_to_expiry
                    })
                    
                    # Add put option
                    synthetic_data.append({
                        'symbol': f"{symbol}_{expiry_date.replace('-', '')}P{int(strike)}",
                        'underlying': symbol,
                        'strike': strike,
                        'option_type': 'PUT',
                        'expiration_date': expiry_date,
                        'bid': put_price * 0.98,  # Simulate bid-ask spread
                        'ask': put_price * 1.02,
                        'last': put_price,
                        'delta': put_delta,
                        'gamma': put_gamma,
                        'theta': put_theta,
                        'vega': put_vega,
                        'implied_volatility': vol,
                        'timestamp': date,
                        'days_to_expiry': days_to_expiry
                    })
        
        logger.info(f"Generated {len(synthetic_data)} synthetic options records")
        return pd.DataFrame(synthetic_data)
    
    def _black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option price using Black-Scholes formula"""
        try:
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
        """Calculate option Greeks using Black-Scholes formula"""
        try:
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
