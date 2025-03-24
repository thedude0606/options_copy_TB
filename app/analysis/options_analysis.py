"""
Options analysis module for options recommendation platform.
Implements options Greeks calculations and analysis.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import math

class OptionsAnalysis:
    """
    Class to analyze options data and calculate Greeks
    """
    
    @staticmethod
    def calculate_black_scholes(S, K, T, r, sigma, option_type='call'):
        """
        Calculate option price using Black-Scholes model
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Option price
        """
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_delta(S, K, T, r, sigma, option_type='call'):
        """
        Calculate Delta (first derivative of option price with respect to underlying price)
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Delta value
        """
        if T <= 0 or sigma <= 0:
            return 1 if option_type.lower() == 'call' and S > K else 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        return delta
    
    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        """
        Calculate Gamma (second derivative of option price with respect to underlying price)
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            
        Returns:
            float: Gamma value
        """
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return gamma
    
    @staticmethod
    def calculate_theta(S, K, T, r, sigma, option_type='call'):
        """
        Calculate Theta (derivative of option price with respect to time)
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Theta value (per day)
        """
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate theta (per year)
        if option_type.lower() == 'call':
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert to daily theta
        theta = theta / 365
        
        return theta
    
    @staticmethod
    def calculate_vega(S, K, T, r, sigma):
        """
        Calculate Vega (derivative of option price with respect to volatility)
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            
        Returns:
            float: Vega value (for 1% change in volatility)
        """
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        # Calculate vega (for 1 point change in volatility)
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        # Convert to vega for 1% change in volatility
        vega = vega / 100
        
        return vega
    
    @staticmethod
    def calculate_rho(S, K, T, r, sigma, option_type='call'):
        """
        Calculate Rho (derivative of option price with respect to interest rate)
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Rho value (for 1% change in interest rate)
        """
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate rho (for 1 point change in interest rate)
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert to rho for 1% change in interest rate
        rho = rho / 100
        
        return rho
    
    @staticmethod
    def calculate_implied_volatility(option_price, S, K, T, r, option_type='call', precision=0.00001, max_iterations=100):
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            option_price (float): Market price of the option
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            option_type (str): 'call' or 'put'
            precision (float): Desired precision
            max_iterations (int): Maximum number of iterations
            
        Returns:
            float: Implied volatility
        """
        if T <= 0 or option_price <= 0:
            return 0
        
        # Initial guess for volatility
        sigma = 0.3
        
        for i in range(max_iterations):
            # Calculate option price with current sigma
            price = OptionsAnalysis.calculate_black_scholes(S, K, T, r, sigma, option_type)
            
            # Calculate vega
            vega = OptionsAnalysis.calculate_vega(S, K, T, r, sigma) * 100  # Convert back to raw vega
            
            # If vega is too small, avoid division by zero
            if abs(vega) < 1e-10:
                return sigma
            
            # Calculate difference
            diff = option_price - price
            
            # Check if precision is reached
            if abs(diff) < precision:
                return sigma
            
            # Update sigma using Newton-Raphson
            sigma = sigma + diff / vega
            
            # Ensure sigma is positive
            if sigma <= 0:
                sigma = 0.001
        
        return sigma
    
    @staticmethod
    def calculate_all_greeks(options_data, risk_free_rate=0.05):
        """
        Calculate all Greeks for options data
        
        Args:
            options_data (pd.DataFrame): Options data with required fields
            risk_free_rate (float): Risk-free interest rate
            
        Returns:
            pd.DataFrame: Options data with calculated Greeks
        """
        if options_data.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result = options_data.copy()
        
        # Check if required columns exist
        required_cols = ['strikePrice', 'expirationDate', 'optionType', 'last', 'volatility']
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return result
        
        # Convert expiration date to time to expiration in years
        if 'expirationDate' in result.columns:
            # Assuming expirationDate is in format 'YYYY-MM-DD'
            result['daysToExpiration'] = pd.to_datetime(result['expirationDate']) - pd.Timestamp.now()
            result['timeToExpiration'] = result['daysToExpiration'].dt.days / 365
        
        # Get underlying price
        if 'underlyingPrice' in result.columns:
            S = result['underlyingPrice']
        elif 'underlying' in result.columns:
            S = result['underlying']
        else:
            # If no underlying price column, try to use a single value if available
            if 'underlyingPrice' in options_data.iloc[0]:
                S = pd.Series([options_data.iloc[0]['underlyingPrice']] * len(result))
            else:
                print("No underlying price available")
                return result
        
        # Calculate Greeks
        for idx, row in result.iterrows():
            try:
                # Extract parameters
                K = row['strikePrice']
                T = max(row['timeToExpiration'], 0.00001)  # Avoid division by zero
                r = risk_free_rate
                sigma = row['volatility'] / 100 if row['volatility'] > 1 else row['volatility']  # Convert to decimal if needed
                option_type = row['optionType'].lower()
                
                # Calculate Greeks
                result.loc[idx, 'delta'] = OptionsAnalysis.calculate_delta(S[idx], K, T, r, sigma, option_type)
                result.loc[idx, 'gamma'] = OptionsAnalysis.calculate_gamma(S[idx], K, T, r, sigma)
                result.loc[idx, 'theta'] = OptionsAnalysis.calculate_theta(S[idx], K, T, r, sigma, option_type)
                result.loc[idx, 'vega'] = OptionsAnalysis.calculate_vega(S[idx], K, T, r, sigma)
                result.loc[idx, 'rho'] = OptionsAnalysis.calculate_rho(S[idx], K, T, r, sigma, option_type)
                
                # Calculate implied volatility if not provided
                if pd.isna(row['volatility']) and 'last' in row and row['last'] > 0:
                    result.loc[idx, 'impliedVolatility'] = OptionsAnalysis.calculate_implied_volatility(
                        row['last'], S[idx], K, T, r, option_type
                    )
            except Exception as e:
                print(f"Error calculating Greeks for row {idx}: {str(e)}")
        
        return result
    
    @staticmethod
    def calculate_probability_of_profit(options_data, risk_free_rate=0.05):
        """
        Calculate probability of profit for options
        
        Args:
            options_data (pd.DataFrame): Options data with required fields
            risk_free_rate (float): Risk-free interest rate
            
        Returns:
            pd.DataFrame: Options data with probability of profit
        """
        if options_data.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result = options_data.copy()
        
        # Check if required columns exist
        required_cols = ['strikePrice', 'expirationDate', 'optionType', 'volatility']
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return result
        
        # Convert expiration date to time to expiration in years
        if 'expirationDate' in result.columns:
            # Assuming expirationDate is in format 'YYYY-MM-DD'
            result['daysToExpiration'] = pd.to_datetime(result['expirationDate']) - pd.Timestamp.now()
            result['timeToExpiration'] = result['daysToExpiration'].dt.days / 365
        
        # Get underlying price
        if 'underlyingPrice' in result.columns:
            S = result['underlyingPrice']
        elif 'underlying' in result.columns:
            S = result['underlying']
        else:
            # If no underlying price column, try to use a single value if available
            if 'underlyingPrice' in options_data.iloc[0]:
                S = pd.Series([options_data.iloc[0]['underlyingPrice']] * len(result))
            else:
                print("No underlying price available")
                return result
        
        # Calculate probability of profit
        for idx, row in result.iterrows():
            try:
                # Extract parameters
                K = row['strikePrice']
                T = max(row['timeToExpiration'], 0.00001)  # Avoid division by zero
                r = risk_free_rate
                sigma = row['volatility'] / 100 if row['volatility'] > 1 else row['volatility']  # Convert to decimal if needed
                option_type = row['optionType'].lower()
                
                # For long options
                if option_type == 'call':
                    # Probability that stock price will be above strike at expiration
                    d2 = (np.log(S[idx] / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    prob_profit = norm.cdf(d2)
                else:  # put
                    # Probability that stock price will be below strike at expiration
                    d2 = (np.log(S[idx] / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    prob_profit = norm.cdf(-d2)
                
                result.loc[idx, 'probabilityOfProfit'] = prob_profit
                
                # Calculate expected profit
                option_price = row['last'] if 'last' in row else row['mark'] if 'mark' in row else 0
                if option_price > 0:
                    if option_type == 'call':
                        expected_profit = prob_profit * (S[idx] * np.exp(r * T) - K) - option_price
                    else:  # put
                        expected_profit = prob_profit * (K - S[idx] * np.exp(r * T)) - option_price
                    
                    result.loc[idx, 'expectedProfit'] = expected_profit
            except Exception as e:
                print(f"Error calculating probability of profit for row {idx}: {str(e)}")
        
        return result
    
    @staticmethod
    def calculate_risk_reward_ratio(options_data, target_price=None):
        """
        Calculate risk-reward ratio for options
        
        Args:
            options_data (pd.DataFrame): Options data with required fields
            target_price (float): Target price for the underlying asset
            
        Returns:
            pd.DataFrame: Options data with risk-reward ratio
        """
        if options_data.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result = options_data.copy()
        
        # Check if required columns exist
        required_cols = ['strikePrice', 'optionType', 'last']
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return result
        
        # Get underlying price
        if 'underlyingPrice' in result.columns:
            S = result['underlyingPrice']
        elif 'underlying' in result.columns:
            S = result['underlying']
        else:
            # If no underlying price column, try to use a single value if available
            if 'underlyingPrice' in options_data.iloc[0]:
                S = pd.Series([options_data.iloc[0]['underlyingPrice']] * len(result))
            else:
                print("No underlying price available")
                return result
        
        # Use target price if provided, otherwise use 20% move
        if target_price is None:
            target_prices = S * 1.2  # 20% upside for calls
        else:
            target_prices = pd.Series([target_price] * len(result))
        
        # Calculate risk-reward ratio
        for idx, row in result.iterrows():
            try:
                # Extract parameters
                K = row['strikePrice']
                option_type = row['optionType'].lower()
                option_price = row['last'] if 'last' in row else row['mark'] if 'mark' in row else 0
                
                if option_price <= 0:
                    continue
                
                # Calculate potential reward and risk
                if option_type == 'call':
                    # Potential reward: target price - strike (if positive)
                    potential_reward = max(0, target_prices[idx] - K)
                    # Risk: option price paid
                    risk = option_price
                else:  # put
                    # Potential reward: strike - target price (if positive)
                    potential_reward = max(0, K - target_prices[idx])
                    # Risk: option price paid
                    risk = option_price
                
                # Calculate risk-reward ratio
                if risk > 0 and potential_reward > 0:
                    risk_reward_ratio = potential_reward / risk
                    result.loc[idx, 'riskRewardRatio'] = risk_reward_ratio
                    
                    # Calculate percentage returns
                    percent_return = (potential_reward - risk) / risk * 100
                    result.loc[idx, 'potentialReturnPercent'] = percent_return
            except Exception as e:
                print(f"Error calculating risk-reward ratio for row {idx}: {str(e)}")
        
        return result
