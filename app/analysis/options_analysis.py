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
            
            # Create a numeric days column to avoid .dt accessor issues
            def get_days(x):
                if isinstance(x, pd.Timedelta):
                    return x.days
                elif pd.isna(x):
                    return 0
                else:
                    return float(x)
            
            # Apply the function to create a numeric days column
            result['days_numeric'] = result['daysToExpiration'].apply(get_days)
            result['timeToExpiration'] = result['days_numeric'] / 365
        
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
                
                # Use days_numeric instead of .dt.days
                if 'days_numeric' in row and not pd.isna(row['days_numeric']):
                    T = row['days_numeric'] / 365  # Convert days to years
                else:
                    T = 0.01  # Default to a small positive value
                
                if T <= 0:
                    T = 0.01  # Avoid division by zero
                
                sigma = row['volatility'] / 100 if 'volatility' in row else 0.3  # Convert percentage to decimal
                r = risk_free_rate
                option_type = row['optionType'].lower() if 'optionType' in row else 'call'
                
                # Calculate Greeks
                result.at[idx, 'delta'] = OptionsAnalysis.calculate_delta(S.iloc[idx], K, T, r, sigma, option_type)
                result.at[idx, 'gamma'] = OptionsAnalysis.calculate_gamma(S.iloc[idx], K, T, r, sigma)
                result.at[idx, 'theta'] = OptionsAnalysis.calculate_theta(S.iloc[idx], K, T, r, sigma, option_type)
                result.at[idx, 'vega'] = OptionsAnalysis.calculate_vega(S.iloc[idx], K, T, r, sigma)
                result.at[idx, 'rho'] = OptionsAnalysis.calculate_rho(S.iloc[idx], K, T, r, sigma, option_type)
                
                # Calculate implied volatility if market price is available
                if 'last' in row and row['last'] > 0:
                    result.at[idx, 'impliedVolatility'] = OptionsAnalysis.calculate_implied_volatility(
                        row['last'], S.iloc[idx], K, T, r, option_type
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
        required_cols = ['strikePrice', 'expirationDate', 'optionType', 'last', 'underlyingPrice']
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return result
        
        # Convert expiration date to time to expiration in years
        if 'expirationDate' in result.columns:
            # Assuming expirationDate is in format 'YYYY-MM-DD'
            result['daysToExpiration'] = pd.to_datetime(result['expirationDate']) - pd.Timestamp.now()
            
            # Create a numeric days column to avoid .dt accessor issues
            def get_days(x):
                if isinstance(x, pd.Timedelta):
                    return x.days
                elif pd.isna(x):
                    return 0
                else:
                    return float(x)
            
            # Apply the function to create a numeric days column
            result['days_numeric'] = result['daysToExpiration'].apply(get_days)
            result['timeToExpiration'] = result['days_numeric'] / 365
        
        # Calculate probability of profit
        for idx, row in result.iterrows():
            try:
                # Extract parameters
                S = row['underlyingPrice']
                K = row['strikePrice']
                
                # Use days_numeric instead of .dt.days
                if 'days_numeric' in row and not pd.isna(row['days_numeric']):
                    T = row['days_numeric'] / 365  # Convert days to years
                else:
                    T = 0.01  # Default to a small positive value
                
                if T <= 0:
                    T = 0.01  # Avoid division by zero
                
                # Use implied volatility if available, otherwise use historical volatility
                if 'impliedVolatility' in row and not pd.isna(row['impliedVolatility']):
                    sigma = row['impliedVolatility']
                elif 'volatility' in row and not pd.isna(row['volatility']):
                    sigma = row['volatility'] / 100  # Convert percentage to decimal
                else:
                    sigma = 0.3  # Default volatility
                
                option_type = row['optionType'].lower() if 'optionType' in row else 'call'
                option_price = row['last'] if 'last' in row else 0
                
                # Calculate probability of profit
                if option_type == 'call':
                    # For long call, stock price needs to be above strike + premium
                    breakeven = K + option_price
                    # Calculate probability that stock will be above breakeven at expiration
                    d = (np.log(S / breakeven) + (risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    pop = norm.cdf(d)
                else:  # put
                    # For long put, stock price needs to be below strike - premium
                    breakeven = K - option_price
                    # Calculate probability that stock will be below breakeven at expiration
                    d = (np.log(S / breakeven) + (risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    pop = 1 - norm.cdf(d)
                
                result.at[idx, 'probabilityOfProfit'] = pop
                result.at[idx, 'breakeven'] = breakeven
            except Exception as e:
                print(f"Error calculating probability of profit for row {idx}: {str(e)}")
        
        return result
    
    @staticmethod
    def calculate_risk_reward_ratio(options_data):
        """
        Calculate risk-reward ratio for options
        
        Args:
            options_data (pd.DataFrame): Options data with required fields
            
        Returns:
            pd.DataFrame: Options data with risk-reward ratio
        """
        if options_data.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result = options_data.copy()
        
        # Calculate risk-reward ratio
        for idx, row in result.iterrows():
            try:
                option_type = row['optionType'].lower() if 'optionType' in row else 'call'
                option_price = row['last'] if 'last' in row else 0
                
                if option_price <= 0:
                    result.at[idx, 'riskRewardRatio'] = 0
                    result.at[idx, 'potentialReturn'] = 0
                    continue
                
                # Calculate maximum potential profit and loss
                if option_type == 'call':
                    # For long call, max loss is premium paid
                    max_loss = option_price
                    # Max profit is theoretically unlimited, but we'll use a reasonable estimate
                    # based on implied volatility and time to expiration
                    if 'impliedVolatility' in row and not pd.isna(row['impliedVolatility']) and 'timeToExpiration' in row:
                        # Estimate potential upside move based on volatility
                        sigma = row['impliedVolatility']
                        T = row['timeToExpiration']
                        S = row['underlyingPrice']
                        K = row['strikePrice']
                        
                        # Estimate potential price move (1 standard deviation)
                        potential_move = S * sigma * np.sqrt(T)
                        potential_price = S + potential_move
                        
                        # Calculate potential profit
                        potential_profit = max(0, potential_price - K) - option_price
                        max_profit = potential_profit
                    else:
                        # Simple estimate: 2x the premium
                        max_profit = option_price * 2
                else:  # put
                    # For long put, max loss is premium paid
                    max_loss = option_price
                    # Max profit is strike price - premium (if stock goes to zero)
                    K = row['strikePrice']
                    max_profit = max(0, K - option_price)
                
                # Calculate risk-reward ratio
                if max_loss > 0:
                    risk_reward = max_profit / max_loss
                else:
                    risk_reward = 0
                
                # Calculate potential return percentage
                if option_price > 0:
                    potential_return_pct = (max_profit / option_price) * 100
                    print(f"Calculated potential return: {potential_return_pct:.2f}% (Max profit: ${max_profit:.2f}, Option price: ${option_price:.2f})")
                else:
                    potential_return_pct = 0
                
                result.at[idx, 'riskRewardRatio'] = risk_reward
                result.at[idx, 'maxProfit'] = max_profit
                result.at[idx, 'maxLoss'] = max_loss
                result.at[idx, 'potentialReturn'] = potential_return_pct
            except Exception as e:
                print(f"Error calculating risk-reward ratio for row {idx}: {str(e)}")
        
        return result
