"""
Risk Management Components for Options Trading.
Implements position sizing, stop-loss/take-profit recommendations, and portfolio risk management.
"""
import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats
import matplotlib.pyplot as plt
import io
import os

class PositionSizer:
    """
    Position sizing calculator for options trading.
    Determines optimal position size based on risk tolerance and market conditions.
    """
    def __init__(self, account_size=100000, max_risk_pct=2.0, max_position_pct=5.0,
                 volatility_adjustment=True, kelly_criterion=False):
        """
        Initialize the position sizer.
        
        Parameters:
        -----------
        account_size : float
            Total account size in dollars
        max_risk_pct : float
            Maximum risk per trade as percentage of account
        max_position_pct : float
            Maximum position size as percentage of account
        volatility_adjustment : bool
            Whether to adjust position size based on volatility
        kelly_criterion : bool
            Whether to use Kelly criterion for position sizing
        """
        self.account_size = account_size
        self.max_risk_pct = max_risk_pct
        self.max_position_pct = max_position_pct
        self.volatility_adjustment = volatility_adjustment
        self.kelly_criterion = kelly_criterion
    
    def calculate_position_size(self, option_data, confidence_score=None, 
                               win_rate=None, profit_loss_ratio=None):
        """
        Calculate optimal position size for an options trade.
        
        Parameters:
        -----------
        option_data : dict or pandas.Series
            Option data including price, strike, implied volatility, etc.
        confidence_score : float, optional
            Confidence score from recommendation engine (0-1)
        win_rate : float, optional
            Historical win rate for similar trades (0-1)
        profit_loss_ratio : float, optional
            Ratio of average profit to average loss
            
        Returns:
        --------
        dict
            Dictionary with position sizing details
        """
        # Extract option data
        option_price = self._get_option_price(option_data)
        option_type = self._get_option_type(option_data)
        
        # Calculate max position size based on account percentage
        max_position_dollars = self.account_size * (self.max_position_pct / 100)
        
        # Calculate max risk amount
        max_risk_dollars = self.account_size * (self.max_risk_pct / 100)
        
        # Calculate contract multiplier (typically 100 for equity options)
        contract_multiplier = 100
        
        # Calculate max number of contracts based on risk
        max_contracts_by_risk = max_risk_dollars / (option_price * contract_multiplier)
        
        # Calculate max number of contracts based on position size
        max_contracts_by_size = max_position_dollars / (option_price * contract_multiplier)
        
        # Take the minimum of the two constraints
        max_contracts = min(int(max_contracts_by_risk), int(max_contracts_by_size))
        
        # Apply volatility adjustment if enabled
        if self.volatility_adjustment and 'impliedVolatility' in option_data:
            iv = option_data['impliedVolatility']
            # Reduce position size for high volatility
            volatility_factor = self._calculate_volatility_factor(iv)
            volatility_adjusted_contracts = int(max_contracts * volatility_factor)
            max_contracts = min(max_contracts, volatility_adjusted_contracts)
        
        # Apply Kelly criterion if enabled and have win rate and profit/loss ratio
        kelly_contracts = None
        if self.kelly_criterion and win_rate is not None and profit_loss_ratio is not None:
            kelly_fraction = self._calculate_kelly_fraction(win_rate, profit_loss_ratio)
            kelly_contracts = int(self.account_size * kelly_fraction / (option_price * contract_multiplier))
            # Use Kelly as a maximum constraint
            max_contracts = min(max_contracts, kelly_contracts)
        
        # Apply confidence adjustment if provided
        if confidence_score is not None:
            confidence_factor = self._calculate_confidence_factor(confidence_score)
            confidence_adjusted_contracts = int(max_contracts * confidence_factor)
            max_contracts = min(max_contracts, confidence_adjusted_contracts)
        
        # Ensure at least 1 contract if account can afford it
        if max_contracts < 1 and self.account_size >= option_price * contract_multiplier:
            max_contracts = 1
        
        # Calculate total cost and risk
        total_cost = max_contracts * option_price * contract_multiplier
        total_risk = min(total_cost, max_risk_dollars)
        
        # Compile results
        result = {
            'recommended_contracts': max_contracts,
            'contract_price': option_price,
            'total_cost': total_cost,
            'total_risk': total_risk,
            'risk_percentage': (total_risk / self.account_size) * 100,
            'position_percentage': (total_cost / self.account_size) * 100
        }
        
        # Add additional details if available
        if self.volatility_adjustment and 'impliedVolatility' in option_data:
            result['volatility_factor'] = volatility_factor
        
        if self.kelly_criterion and kelly_contracts is not None:
            result['kelly_fraction'] = kelly_fraction
            result['kelly_contracts'] = kelly_contracts
        
        if confidence_score is not None:
            result['confidence_factor'] = confidence_factor
        
        return result
    
    def _get_option_price(self, option_data):
        """
        Extract option price from option data.
        
        Parameters:
        -----------
        option_data : dict or pandas.Series
            Option data
            
        Returns:
        --------
        float
            Option price
        """
        # Try different field names for price
        for field in ['mid_price', 'mark', 'last', 'price']:
            if field in option_data:
                return float(option_data[field])
        
        # If bid and ask are available, use mid price
        if 'bid' in option_data and 'ask' in option_data:
            return (float(option_data['bid']) + float(option_data['ask'])) / 2
        
        # Default fallback
        return float(option_data.get('price', 1.0))
    
    def _get_option_type(self, option_data):
        """
        Extract option type from option data.
        
        Parameters:
        -----------
        option_data : dict or pandas.Series
            Option data
            
        Returns:
        --------
        str
            Option type ('call' or 'put')
        """
        # Try different field names for option type
        for field in ['putCall', 'option_type', 'type']:
            if field in option_data:
                value = str(option_data[field]).upper()
                if 'CALL' in value:
                    return 'call'
                elif 'PUT' in value:
                    return 'put'
        
        # Default fallback
        return 'call'
    
    def _calculate_volatility_factor(self, implied_volatility):
        """
        Calculate position size adjustment factor based on implied volatility.
        
        Parameters:
        -----------
        implied_volatility : float
            Option implied volatility
            
        Returns:
        --------
        float
            Adjustment factor (0-1)
        """
        # Higher IV = smaller position size
        # IV of 0.3 (30%) is considered average
        if implied_volatility <= 0.3:
            return 1.0
        elif implied_volatility <= 0.5:
            return 0.8
        elif implied_volatility <= 0.7:
            return 0.6
        elif implied_volatility <= 1.0:
            return 0.4
        else:
            return 0.25
    
    def _calculate_kelly_fraction(self, win_rate, profit_loss_ratio):
        """
        Calculate Kelly criterion optimal fraction.
        
        Parameters:
        -----------
        win_rate : float
            Probability of winning (0-1)
        profit_loss_ratio : float
            Ratio of average profit to average loss
            
        Returns:
        --------
        float
            Kelly fraction (0-1)
        """
        # Kelly formula: f* = (bp - q) / b
        # where p = win probability, q = loss probability, b = profit/loss ratio
        q = 1 - win_rate
        kelly = (win_rate * profit_loss_ratio - q) / profit_loss_ratio
        
        # Limit Kelly to reasonable range
        kelly = max(0, min(kelly, 0.5))  # Cap at 50% of Kelly
        
        return kelly
    
    def _calculate_confidence_factor(self, confidence_score):
        """
        Calculate position size adjustment factor based on confidence score.
        
        Parameters:
        -----------
        confidence_score : float
            Confidence score (0-1)
            
        Returns:
        --------
        float
            Adjustment factor (0-1)
        """
        # Higher confidence = larger position size
        if confidence_score >= 0.8:
            return 1.0
        elif confidence_score >= 0.6:
            return 0.8
        elif confidence_score >= 0.4:
            return 0.6
        elif confidence_score >= 0.2:
            return 0.4
        else:
            return 0.2


class StopLossTakeProfitCalculator:
    """
    Calculates optimal stop-loss and take-profit levels for options trades.
    """
    def __init__(self, risk_reward_ratio=2.0, use_volatility=True, 
                 use_support_resistance=True, time_based_exits=True):
        """
        Initialize the stop-loss/take-profit calculator.
        
        Parameters:
        -----------
        risk_reward_ratio : float
            Target risk-reward ratio
        use_volatility : bool
            Whether to use volatility for level calculation
        use_support_resistance : bool
            Whether to use support/resistance levels
        time_based_exits : bool
            Whether to calculate time-based exit points
        """
        self.risk_reward_ratio = risk_reward_ratio
        self.use_volatility = use_volatility
        self.use_support_resistance = use_support_resistance
        self.time_based_exits = time_based_exits
    
    def calculate_exit_points(self, option_data, entry_price=None, 
                             support_resistance_levels=None, price_history=None):
        """
        Calculate optimal exit points for an options trade.
        
        Parameters:
        -----------
        option_data : dict or pandas.Series
            Option data including price, strike, implied volatility, etc.
        entry_price : float, optional
            Entry price for the option (defaults to current price)
        support_resistance_levels : dict, optional
            Dictionary with support and resistance levels
        price_history : pandas.DataFrame, optional
            Historical price data for underlying asset
            
        Returns:
        --------
        dict
            Dictionary with exit point details
        """
        # Extract option data
        option_price = entry_price or self._get_option_price(option_data)
        option_type = self._get_option_type(option_data)
        days_to_expiration = self._get_days_to_expiration(option_data)
        
        # Initialize result dictionary
        result = {
            'entry_price': option_price,
            'option_type': option_type,
            'days_to_expiration': days_to_expiration
        }
        
        # Calculate basic stop-loss and take-profit levels
        stop_loss = self._calculate_basic_stop_loss(option_price, option_type)
        take_profit = self._calculate_basic_take_profit(option_price, option_type, stop_loss)
        
        result['basic_stop_loss'] = stop_loss
        result['basic_take_profit'] = take_profit
        
        # Adjust based on volatility if enabled
        if self.use_volatility and 'impliedVolatility' in option_data:
            iv = option_data['impliedVolatility']
            volatility_stop_loss = self._adjust_for_volatility(option_price, iv, 'stop_loss')
            volatility_take_profit = self._adjust_for_volatility(option_price, iv, 'take_profit')
            
            result['volatility_adjusted_stop_loss'] = volatility_stop_loss
            result['volatility_adjusted_take_profit'] = volatility_take_profit
            
            # Update stop-loss and take-profit with volatility adjusted values
            stop_loss = volatility_stop_loss
            take_profit = volatility_take_profit
        
        # Adjust based on support/resistance if enabled
        if self.use_support_resistance and support_resistance_levels is not None:
            sr_stop_loss, sr_take_profit = self._adjust_for_support_resistance(
                option_price, option_type, support_resistance_levels
            )
            
            result['sr_adjusted_stop_loss'] = sr_stop_loss
            result['sr_adjusted_take_profit'] = sr_take_profit
            
            # Update stop-loss and take-profit with support/resistance adjusted values
            if sr_stop_loss is not None:
                stop_loss = sr_stop_loss
            if sr_take_profit is not None:
                take_profit = sr_take_profit
        
        # Calculate time-based exits if enabled
        if self.time_based_exits and days_to_expiration is not None:
            time_exits = self._calculate_time_based_exits(days_to_expiration, option_data)
            result['time_based_exits'] = time_exits
        
        # Calculate risk-reward ratio
        risk = option_price - stop_loss
        reward = take_profit - option_price
        risk_reward = reward / risk if risk > 0 else 0
        
        result['final_stop_loss'] = stop_loss
        result['final_take_profit'] = take_profit
        result['risk_amount'] = risk
        result['reward_amount'] = reward
        result['risk_reward_ratio'] = risk_reward
        result['risk_percentage'] = (risk / option_price) * 100
        result['reward_percentage'] = (reward / option_price) * 100
        
        return result
    
    def _get_option_price(self, option_data):
        """
        Extract option price from option data.
        
        Parameters:
        -----------
        option_data : dict or pandas.Series
            Option data
            
        Returns:
        --------
        float
            Option price
        """
        # Try different field names for price
        for field in ['mid_price', 'mark', 'last', 'price']:
            if field in option_data:
                return float(option_data[field])
        
        # If bid and ask are available, use mid price
        if 'bid' in option_data and 'ask' in option_data:
            return (float(option_data['bid']) + float(option_data['ask'])) / 2
        
        # Default fallback
        return float(option_data.get('price', 1.0))
    
    def _get_option_type(self, option_data):
        """
        Extract option type from option data.
        
        Parameters:
        -----------
        option_data : dict or pandas.Series
            Option data
            
        Returns:
        --------
        str
            Option type ('call' or 'put')
        """
        # Try different field names for option type
        for field in ['putCall', 'option_type', 'type']:
            if field in option_data:
                value = str(option_data[field]).upper()
                if 'CALL' in value:
                    return 'call'
                elif 'PUT' in value:
                    return 'put'
        
        # Default fallback
        return 'call'
    
    def _get_days_to_expiration(self, option_data):
        """
        Extract days to expiration from option data.
        
        Parameters:
        -----------
        option_data : dict or pandas.Series
            Option data
            
        Returns:
        --------
        int
            Days to expiration
        """
        # Try direct days to expiration field
        if 'daysToExpiration' in option_data:
            return int(option_data['daysToExpiration'])
        
        # Try expiration date field
        for field in ['expirationDate', 'expiration']:
            if field in option_data:
                try:
                    exp_date = pd.to_datetime(option_data[field])
                    today = pd.Timestamp.today().normalize()
                    return max(0, (exp_date - today).days)
                except:
                    pass
        
        # Default fallback
        return 30  # Assume 30 days if not specified
    
    def _calculate_basic_stop_loss(self, option_price, option_type):
        """
        Calculate basic stop-loss level.
        
        Parameters:
        -----------
        option_price : float
            Current option price
        option_type : str
            Option type ('call' or 'put')
            
        Returns:
        --------
        float
            Stop-loss price
        """
        # Basic stop-loss is typically 50% of option price
        return option_price * 0.5
    
    def _calculate_basic_take_profit(self, option_price, option_type, stop_loss):
        """
        Calculate basic take-profit level.
        
        Parameters:
        -----------
        option_price : float
            Current option price
        option_type : str
            Option type ('call' or 'put')
        stop_loss : float
            Stop-loss price
            
        Returns:
        --------
        float
            Take-profit price
        """
        # Calculate risk amount
        risk = option_price - stop_loss
        
        # Calculate take-profit based on risk-reward ratio
        take_profit = option_price + (risk * self.risk_reward_ratio)
        
        return take_profit
    
    def _adjust_for_volatility(self, option_price, implied_volatility, level_type):
        """
        Adjust stop-loss or take-profit level based on implied volatility.
        
        Parameters:
        -----------
        option_price : float
            Current option price
        implied_volatility : float
            Option implied volatility
        level_type : str
            Type of level to adjust ('stop_loss' or 'take_profit')
            
        Returns:
        --------
        float
            Adjusted price level
        """
        # Higher IV = wider stops and targets
        if level_type == 'stop_loss':
            # For stop-loss, higher IV means lower stop (wider)
            if implied_volatility <= 0.3:
                return option_price * 0.6  # 40% drop
            elif implied_volatility <= 0.5:
                return option_price * 0.5  # 50% drop
            elif implied_volatility <= 0.7:
                return option_price * 0.4  # 60% drop
            else:
                return option_price * 0.3  # 70% drop
        else:  # take_profit
            # For take-profit, higher IV means higher target (wider)
            if implied_volatility <= 0.3:
                return option_price * 1.8  # 80% gain
            elif implied_volatility <= 0.5:
                return option_price * 2.0  # 100% gain
            elif implied_volatility <= 0.7:
                return option_price * 2.5  # 150% gain
            else:
                return option_price * 3.0  # 200% gain
    
    def _adjust_for_support_resistance(self, option_price, option_type, sr_levels):
        """
        Adjust stop-loss and take-profit levels based on support/resistance.
        
        Parameters:
        -----------
        option_price : float
            Current option price
        option_type : str
            Option type ('call' or 'put')
        sr_levels : dict
            Dictionary with support and resistance levels
            
        Returns:
        --------
        tuple
            (adjusted_stop_loss, adjusted_take_profit)
        """
        # Initialize with None (no adjustment)
        adjusted_stop_loss = None
        adjusted_take_profit = None
        
        # Check if we have the necessary data
        if not sr_levels or not isinstance(sr_levels, dict):
            return adjusted_stop_loss, adjusted_take_profit
        
        # Extract support and resistance levels
        supports = sr_levels.get('support', [])
        resistances = sr_levels.get('resistance', [])
        
        if not supports or not resistances:
            return adjusted_stop_loss, adjusted_take_profit
        
        # Sort levels
        supports = sorted(supports)
        resistances = sorted(resistances)
        
        # For calls, use nearest support for stop-loss and nearest resistance for take-profit
        if option_type == 'call':
            # Find nearest support below current price
            nearest_support = None
            for level in reversed(supports):
                if level < option_price:
                    nearest_support = level
                    break
            
            # Find nearest resistance above current price
            nearest_resistance = None
            for level in resistances:
                if level > option_price:
                    nearest_resistance = level
                    break
            
            # Set adjusted levels if found
            if nearest_support is not None:
                adjusted_stop_loss = nearest_support
            
            if nearest_resistance is not None:
                adjusted_take_profit = nearest_resistance
        
        # For puts, use nearest resistance for stop-loss and nearest support for take-profit
        else:
            # Find nearest resistance above current price
            nearest_resistance = None
            for level in resistances:
                if level > option_price:
                    nearest_resistance = level
                    break
            
            # Find nearest support below current price
            nearest_support = None
            for level in reversed(supports):
                if level < option_price:
                    nearest_support = level
                    break
            
            # Set adjusted levels if found
            if nearest_resistance is not None:
                adjusted_stop_loss = nearest_resistance
            
            if nearest_support is not None:
                adjusted_take_profit = nearest_support
        
        return adjusted_stop_loss, adjusted_take_profit
    
    def _calculate_time_based_exits(self, days_to_expiration, option_data):
        """
        Calculate time-based exit points.
        
        Parameters:
        -----------
        days_to_expiration : int
            Days to expiration
        option_data : dict or pandas.Series
            Option data
            
        Returns:
        --------
        dict
            Dictionary with time-based exit details
        """
        # Initialize result
        time_exits = {}
        
        # Calculate days thresholds based on total days to expiration
        if days_to_expiration <= 7:  # Weekly options
            time_exits['exit_days_before_exp'] = 1
            time_exits['partial_exit_days'] = [3, 1]
        elif days_to_expiration <= 30:  # Monthly options
            time_exits['exit_days_before_exp'] = 5
            time_exits['partial_exit_days'] = [14, 7, 3]
        elif days_to_expiration <= 60:  # Two-month options
            time_exits['exit_days_before_exp'] = 10
            time_exits['partial_exit_days'] = [30, 14, 7]
        else:  # Longer-term options
            time_exits['exit_days_before_exp'] = 21
            time_exits['partial_exit_days'] = [days_to_expiration // 2, 21, 14, 7]
        
        # Calculate percentage thresholds
        time_exits['exit_pct_of_time'] = 0.8  # Exit when 80% of time has passed
        
        # Calculate theta decay acceleration point
        # Typically around 30-45 days for most options
        theta_acceleration = min(30, days_to_expiration // 2)
        time_exits['theta_acceleration_point'] = theta_acceleration
        
        # Add calendar dates
        today = pd.Timestamp.today().normalize()
        
        # Final exit date
        final_exit_date = today + pd.Timedelta(days=days_to_expiration - time_exits['exit_days_before_exp'])
        time_exits['final_exit_date'] = final_exit_date.strftime('%Y-%m-%d')
        
        # Partial exit dates
        partial_exit_dates = []
        for days in time_exits['partial_exit_days']:
            if days < days_to_expiration:
                exit_date = today + pd.Timedelta(days=days_to_expiration - days)
                partial_exit_dates.append(exit_date.strftime('%Y-%m-%d'))
        
        time_exits['partial_exit_dates'] = partial_exit_dates
        
        return time_exits


class PortfolioRiskManager:
    """
    Manages risk at the portfolio level for options trading.
    """
    def __init__(self, max_portfolio_risk=15.0, max_sector_risk=25.0, 
                 max_strategy_risk=40.0, correlation_threshold=0.7):
        """
        Initialize the portfolio risk manager.
        
        Parameters:
        -----------
        max_portfolio_risk : float
            Maximum portfolio risk as percentage of account
        max_sector_risk : float
            Maximum risk allocation to a single sector
        max_strategy_risk : float
            Maximum risk allocation to a single strategy type
        correlation_threshold : float
            Correlation threshold for diversification warning
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_sector_risk = max_sector_risk
        self.max_strategy_risk = max_strategy_risk
        self.correlation_threshold = correlation_threshold
        self.positions = []
    
    def add_position(self, position):
        """
        Add a position to the portfolio.
        
        Parameters:
        -----------
        position : dict
            Position details including symbol, type, size, risk, etc.
            
        Returns:
        --------
        int
            Position ID
        """
        # Generate position ID
        position_id = len(self.positions) + 1
        
        # Add ID to position
        position['id'] = position_id
        
        # Add position to portfolio
        self.positions.append(position)
        
        return position_id
    
    def remove_position(self, position_id):
        """
        Remove a position from the portfolio.
        
        Parameters:
        -----------
        position_id : int
            Position ID to remove
            
        Returns:
        --------
        bool
            Whether the position was removed
        """
        # Find position by ID
        for i, position in enumerate(self.positions):
            if position.get('id') == position_id:
                # Remove position
                self.positions.pop(i)
                return True
        
        return False
    
    def update_position(self, position_id, updates):
        """
        Update a position in the portfolio.
        
        Parameters:
        -----------
        position_id : int
            Position ID to update
        updates : dict
            Position updates
            
        Returns:
        --------
        bool
            Whether the position was updated
        """
        # Find position by ID
        for i, position in enumerate(self.positions):
            if position.get('id') == position_id:
                # Update position
                self.positions[i].update(updates)
                return True
        
        return False
    
    def calculate_portfolio_risk(self):
        """
        Calculate overall portfolio risk metrics.
        
        Returns:
        --------
        dict
            Dictionary with portfolio risk metrics
        """
        # Initialize result
        result = {
            'total_positions': len(self.positions),
            'total_exposure': 0,
            'total_risk': 0,
            'sector_exposure': {},
            'sector_risk': {},
            'strategy_exposure': {},
            'strategy_risk': {},
            'risk_concentration': {},
            'diversification_score': 0,
            'correlation_warnings': []
        }
        
        # Calculate total exposure and risk
        for position in self.positions:
            # Extract position details
            exposure = position.get('exposure', 0)
            risk = position.get('risk', 0)
            sector = position.get('sector', 'Unknown')
            strategy = position.get('strategy', 'Unknown')
            
            # Update totals
            result['total_exposure'] += exposure
            result['total_risk'] += risk
            
            # Update sector metrics
            if sector not in result['sector_exposure']:
                result['sector_exposure'][sector] = 0
                result['sector_risk'][sector] = 0
            
            result['sector_exposure'][sector] += exposure
            result['sector_risk'][sector] += risk
            
            # Update strategy metrics
            if strategy not in result['strategy_exposure']:
                result['strategy_exposure'][strategy] = 0
                result['strategy_risk'][strategy] = 0
            
            result['strategy_exposure'][strategy] += exposure
            result['strategy_risk'][strategy] += risk
        
        # Calculate risk concentration
        if result['total_risk'] > 0:
            # Calculate sector risk concentration
            for sector, risk in result['sector_risk'].items():
                result['risk_concentration'][f'sector_{sector}'] = (risk / result['total_risk']) * 100
            
            # Calculate strategy risk concentration
            for strategy, risk in result['strategy_risk'].items():
                result['risk_concentration'][f'strategy_{strategy}'] = (risk / result['total_risk']) * 100
        
        # Calculate diversification score (0-100)
        if len(self.positions) > 0:
            # More positions and lower concentration = higher score
            num_positions_score = min(50, len(self.positions) * 5)  # Max 50 points for 10+ positions
            
            # Calculate concentration score
            concentration_score = 0
            if result['risk_concentration']:
                max_concentration = max(result['risk_concentration'].values())
                concentration_score = max(0, 50 - max_concentration)  # Max 50 points for low concentration
            
            result['diversification_score'] = num_positions_score + concentration_score
        
        # Check for correlation warnings
        result['correlation_warnings'] = self._check_correlations()
        
        # Check for risk limit warnings
        result['risk_warnings'] = self._check_risk_limits(result)
        
        return result
    
    def _check_correlations(self):
        """
        Check for high correlations between positions.
        
        Returns:
        --------
        list
            List of correlation warnings
        """
        warnings = []
        
        # Need at least 2 positions to check correlations
        if len(self.positions) < 2:
            return warnings
        
        # Check each pair of positions
        for i, pos1 in enumerate(self.positions):
            for j, pos2 in enumerate(self.positions[i+1:], i+1):
                # Skip if correlation data not available
                if 'correlation' not in pos1 or 'correlation' not in pos2:
                    continue
                
                # Get correlation between positions
                correlation = self._get_position_correlation(pos1, pos2)
                
                # Add warning if correlation is high
                if correlation is not None and abs(correlation) >= self.correlation_threshold:
                    warnings.append({
                        'position1_id': pos1.get('id'),
                        'position2_id': pos2.get('id'),
                        'position1_symbol': pos1.get('symbol'),
                        'position2_symbol': pos2.get('symbol'),
                        'correlation': correlation,
                        'message': f"High correlation ({correlation:.2f}) between {pos1.get('symbol')} and {pos2.get('symbol')}"
                    })
        
        return warnings
    
    def _get_position_correlation(self, pos1, pos2):
        """
        Get correlation between two positions.
        
        Parameters:
        -----------
        pos1 : dict
            First position
        pos2 : dict
            Second position
            
        Returns:
        --------
        float or None
            Correlation coefficient or None if not available
        """
        # Check if direct correlation is available
        symbol1 = pos1.get('symbol')
        symbol2 = pos2.get('symbol')
        
        if symbol1 in pos2.get('correlation', {}) and isinstance(pos2['correlation'], dict):
            return pos2['correlation'][symbol1]
        
        if symbol2 in pos1.get('correlation', {}) and isinstance(pos1['correlation'], dict):
            return pos1['correlation'][symbol2]
        
        # Default fallback
        return None
    
    def _check_risk_limits(self, risk_metrics):
        """
        Check for risk limit violations.
        
        Parameters:
        -----------
        risk_metrics : dict
            Portfolio risk metrics
            
        Returns:
        --------
        list
            List of risk limit warnings
        """
        warnings = []
        
        # Check total portfolio risk
        total_risk_pct = (risk_metrics['total_risk'] / risk_metrics.get('account_size', 100000)) * 100
        if total_risk_pct > self.max_portfolio_risk:
            warnings.append({
                'type': 'portfolio_risk',
                'current': total_risk_pct,
                'limit': self.max_portfolio_risk,
                'message': f"Portfolio risk ({total_risk_pct:.2f}%) exceeds maximum ({self.max_portfolio_risk:.2f}%)"
            })
        
        # Check sector risk
        for sector, risk in risk_metrics['sector_risk'].items():
            sector_risk_pct = (risk / risk_metrics['total_risk']) * 100 if risk_metrics['total_risk'] > 0 else 0
            if sector_risk_pct > self.max_sector_risk:
                warnings.append({
                    'type': 'sector_risk',
                    'sector': sector,
                    'current': sector_risk_pct,
                    'limit': self.max_sector_risk,
                    'message': f"Risk allocation to {sector} sector ({sector_risk_pct:.2f}%) exceeds maximum ({self.max_sector_risk:.2f}%)"
                })
        
        # Check strategy risk
        for strategy, risk in risk_metrics['strategy_risk'].items():
            strategy_risk_pct = (risk / risk_metrics['total_risk']) * 100 if risk_metrics['total_risk'] > 0 else 0
            if strategy_risk_pct > self.max_strategy_risk:
                warnings.append({
                    'type': 'strategy_risk',
                    'strategy': strategy,
                    'current': strategy_risk_pct,
                    'limit': self.max_strategy_risk,
                    'message': f"Risk allocation to {strategy} strategy ({strategy_risk_pct:.2f}%) exceeds maximum ({self.max_strategy_risk:.2f}%)"
                })
        
        return warnings
    
    def generate_risk_report(self, account_size=None):
        """
        Generate a comprehensive risk report for the portfolio.
        
        Parameters:
        -----------
        account_size : float, optional
            Account size for percentage calculations
            
        Returns:
        --------
        dict
            Dictionary with risk report details
        """
        # Calculate portfolio risk metrics
        risk_metrics = self.calculate_portfolio_risk()
        
        # Add account size if provided
        if account_size is not None:
            risk_metrics['account_size'] = account_size
        
        # Calculate risk as percentage of account
        if 'account_size' in risk_metrics and risk_metrics['account_size'] > 0:
            risk_metrics['total_risk_pct'] = (risk_metrics['total_risk'] / risk_metrics['account_size']) * 100
            risk_metrics['total_exposure_pct'] = (risk_metrics['total_exposure'] / risk_metrics['account_size']) * 100
        
        # Add position details
        risk_metrics['positions'] = self.positions
        
        # Add risk-adjusted return metrics
        risk_metrics['risk_adjusted_metrics'] = self._calculate_risk_adjusted_metrics()
        
        # Add portfolio statistics
        risk_metrics['portfolio_stats'] = self._calculate_portfolio_statistics()
        
        # Add risk recommendations
        risk_metrics['recommendations'] = self._generate_risk_recommendations(risk_metrics)
        
        return risk_metrics
    
    def _calculate_risk_adjusted_metrics(self):
        """
        Calculate risk-adjusted return metrics for the portfolio.
        
        Returns:
        --------
        dict
            Dictionary with risk-adjusted metrics
        """
        # Initialize result
        result = {
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'calmar_ratio': None,
            'profit_factor': None,
            'win_rate': None
        }
        
        # Need positions with return data
        positions_with_returns = [p for p in self.positions if 'expected_return' in p]
        if not positions_with_returns:
            return result
        
        # Calculate expected portfolio return
        total_exposure = sum(p.get('exposure', 0) for p in positions_with_returns)
        if total_exposure <= 0:
            return result
        
        weighted_returns = [
            p.get('expected_return', 0) * (p.get('exposure', 0) / total_exposure)
            for p in positions_with_returns
        ]
        portfolio_return = sum(weighted_returns)
        
        # Calculate risk-free rate (assume 2%)
        risk_free_rate = 0.02
        
        # Calculate Sharpe ratio
        portfolio_risk = sum(p.get('risk', 0) for p in positions_with_returns) / total_exposure
        if portfolio_risk > 0:
            result['sharpe_ratio'] = (portfolio_return - risk_free_rate) / portfolio_risk
        
        # Calculate profit factor
        total_profit = sum(max(0, p.get('expected_return', 0) * p.get('exposure', 0)) for p in positions_with_returns)
        total_loss = sum(max(0, -p.get('expected_return', 0) * p.get('exposure', 0)) for p in positions_with_returns)
        if total_loss > 0:
            result['profit_factor'] = total_profit / total_loss
        
        # Calculate win rate
        winning_positions = sum(1 for p in positions_with_returns if p.get('expected_return', 0) > 0)
        result['win_rate'] = winning_positions / len(positions_with_returns) if positions_with_returns else 0
        
        return result
    
    def _calculate_portfolio_statistics(self):
        """
        Calculate statistical metrics for the portfolio.
        
        Returns:
        --------
        dict
            Dictionary with portfolio statistics
        """
        # Initialize result
        result = {
            'position_count': len(self.positions),
            'avg_position_size': 0,
            'avg_risk_per_position': 0,
            'position_size_distribution': {},
            'risk_distribution': {},
            'sector_distribution': {},
            'strategy_distribution': {}
        }
        
        # Need at least one position
        if not self.positions:
            return result
        
        # Calculate average position size and risk
        total_exposure = sum(p.get('exposure', 0) for p in self.positions)
        total_risk = sum(p.get('risk', 0) for p in self.positions)
        
        result['avg_position_size'] = total_exposure / len(self.positions)
        result['avg_risk_per_position'] = total_risk / len(self.positions)
        
        # Calculate distributions
        sectors = {}
        strategies = {}
        
        for position in self.positions:
            # Extract position details
            exposure = position.get('exposure', 0)
            risk = position.get('risk', 0)
            sector = position.get('sector', 'Unknown')
            strategy = position.get('strategy', 'Unknown')
            
            # Update sector distribution
            if sector not in sectors:
                sectors[sector] = 0
            sectors[sector] += exposure
            
            # Update strategy distribution
            if strategy not in strategies:
                strategies[strategy] = 0
            strategies[strategy] += exposure
        
        # Calculate percentage distributions
        if total_exposure > 0:
            result['sector_distribution'] = {
                sector: (exposure / total_exposure) * 100
                for sector, exposure in sectors.items()
            }
            
            result['strategy_distribution'] = {
                strategy: (exposure / total_exposure) * 100
                for strategy, exposure in strategies.items()
            }
        
        # Calculate position size distribution
        position_sizes = [p.get('exposure', 0) for p in self.positions]
        result['position_size_distribution'] = self._calculate_distribution(position_sizes)
        
        # Calculate risk distribution
        position_risks = [p.get('risk', 0) for p in self.positions]
        result['risk_distribution'] = self._calculate_distribution(position_risks)
        
        return result
    
    def _calculate_distribution(self, values):
        """
        Calculate distribution statistics for a list of values.
        
        Parameters:
        -----------
        values : list
            List of values
            
        Returns:
        --------
        dict
            Dictionary with distribution statistics
        """
        if not values:
            return {}
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'median': sorted(values)[len(values) // 2],
            'std_dev': np.std(values) if len(values) > 1 else 0
        }
    
    def _generate_risk_recommendations(self, risk_metrics):
        """
        Generate risk management recommendations based on portfolio metrics.
        
        Parameters:
        -----------
        risk_metrics : dict
            Portfolio risk metrics
            
        Returns:
        --------
        list
            List of risk management recommendations
        """
        recommendations = []
        
        # Check if portfolio is empty
        if not self.positions:
            recommendations.append({
                'type': 'general',
                'priority': 'high',
                'message': "Portfolio is empty. Consider adding positions based on your strategy and risk tolerance."
            })
            return recommendations
        
        # Check overall portfolio risk
        if 'total_risk_pct' in risk_metrics:
            if risk_metrics['total_risk_pct'] > self.max_portfolio_risk:
                recommendations.append({
                    'type': 'risk_reduction',
                    'priority': 'high',
                    'message': f"Reduce overall portfolio risk from {risk_metrics['total_risk_pct']:.2f}% to below {self.max_portfolio_risk:.2f}%."
                })
            elif risk_metrics['total_risk_pct'] < self.max_portfolio_risk * 0.5:
                recommendations.append({
                    'type': 'risk_increase',
                    'priority': 'medium',
                    'message': f"Portfolio risk ({risk_metrics['total_risk_pct']:.2f}%) is significantly below your maximum ({self.max_portfolio_risk:.2f}%). Consider increasing position sizes or adding new positions."
                })
        
        # Check sector concentration
        for sector, risk_pct in risk_metrics.get('risk_concentration', {}).items():
            if sector.startswith('sector_') and risk_pct > self.max_sector_risk:
                sector_name = sector.replace('sector_', '')
                recommendations.append({
                    'type': 'diversification',
                    'priority': 'high',
                    'message': f"Reduce exposure to {sector_name} sector from {risk_pct:.2f}% to below {self.max_sector_risk:.2f}%."
                })
        
        # Check strategy concentration
        for strategy, risk_pct in risk_metrics.get('risk_concentration', {}).items():
            if strategy.startswith('strategy_') and risk_pct > self.max_strategy_risk:
                strategy_name = strategy.replace('strategy_', '')
                recommendations.append({
                    'type': 'diversification',
                    'priority': 'medium',
                    'message': f"Reduce exposure to {strategy_name} strategy from {risk_pct:.2f}% to below {self.max_strategy_risk:.2f}%."
                })
        
        # Check correlation warnings
        if risk_metrics.get('correlation_warnings'):
            for warning in risk_metrics['correlation_warnings']:
                recommendations.append({
                    'type': 'correlation',
                    'priority': 'medium',
                    'message': warning['message'],
                    'details': {
                        'position1': warning['position1_symbol'],
                        'position2': warning['position2_symbol'],
                        'correlation': warning['correlation']
                    }
                })
        
        # Check diversification score
        if risk_metrics.get('diversification_score', 0) < 50:
            recommendations.append({
                'type': 'diversification',
                'priority': 'medium',
                'message': f"Improve portfolio diversification (current score: {risk_metrics.get('diversification_score', 0)}/100) by adding positions in different sectors or with different strategies."
            })
        
        # Check risk-adjusted metrics
        if risk_metrics.get('risk_adjusted_metrics', {}).get('sharpe_ratio', 0) < 1:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'message': "Consider adjusting your portfolio to improve risk-adjusted returns. Current Sharpe ratio is below 1."
            })
        
        # Check profit factor
        if risk_metrics.get('risk_adjusted_metrics', {}).get('profit_factor', 0) < 1.5:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'message': "Consider adjusting your trading strategy to improve profit factor (ratio of profits to losses)."
            })
        
        return recommendations
    
    def generate_risk_visualization(self, output_dir=None):
        """
        Generate risk visualization charts for the portfolio.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save visualization files
            
        Returns:
        --------
        dict
            Dictionary with visualization file paths or base64 encoded images
        """
        # Calculate portfolio risk metrics
        risk_metrics = self.calculate_portfolio_risk()
        
        # Initialize result
        result = {}
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate sector allocation chart
        sector_chart = self._generate_sector_chart(risk_metrics)
        if sector_chart:
            if output_dir:
                sector_chart_path = os.path.join(output_dir, 'sector_allocation.png')
                sector_chart.savefig(sector_chart_path)
                plt.close()
                result['sector_chart_path'] = sector_chart_path
            else:
                buf = io.BytesIO()
                sector_chart.savefig(buf, format='png')
                buf.seek(0)
                import base64
                result['sector_chart_base64'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
        
        # Generate risk concentration chart
        risk_chart = self._generate_risk_chart(risk_metrics)
        if risk_chart:
            if output_dir:
                risk_chart_path = os.path.join(output_dir, 'risk_concentration.png')
                risk_chart.savefig(risk_chart_path)
                plt.close()
                result['risk_chart_path'] = risk_chart_path
            else:
                buf = io.BytesIO()
                risk_chart.savefig(buf, format='png')
                buf.seek(0)
                import base64
                result['risk_chart_base64'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
        
        # Generate correlation heatmap
        corr_chart = self._generate_correlation_heatmap()
        if corr_chart:
            if output_dir:
                corr_chart_path = os.path.join(output_dir, 'correlation_heatmap.png')
                corr_chart.savefig(corr_chart_path)
                plt.close()
                result['corr_chart_path'] = corr_chart_path
            else:
                buf = io.BytesIO()
                corr_chart.savefig(buf, format='png')
                buf.seek(0)
                import base64
                result['corr_chart_base64'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
        
        return result
    
    def _generate_sector_chart(self, risk_metrics):
        """
        Generate sector allocation chart.
        
        Parameters:
        -----------
        risk_metrics : dict
            Portfolio risk metrics
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            Chart figure or None if not enough data
        """
        # Check if we have sector data
        if not risk_metrics.get('sector_exposure'):
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        sectors = list(risk_metrics['sector_exposure'].keys())
        exposures = list(risk_metrics['sector_exposure'].values())
        
        # Create bar chart
        bars = ax.bar(sectors, exposures)
        
        # Add labels and title
        ax.set_xlabel('Sector')
        ax.set_ylabel('Exposure ($)')
        ax.set_title('Portfolio Exposure by Sector')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', rotation=0)
        
        # Adjust layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def _generate_risk_chart(self, risk_metrics):
        """
        Generate risk concentration chart.
        
        Parameters:
        -----------
        risk_metrics : dict
            Portfolio risk metrics
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            Chart figure or None if not enough data
        """
        # Check if we have risk concentration data
        if not risk_metrics.get('risk_concentration'):
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        categories = []
        values = []
        
        for category, value in risk_metrics['risk_concentration'].items():
            # Clean up category names
            if category.startswith('sector_'):
                category = f"Sector: {category.replace('sector_', '')}"
            elif category.startswith('strategy_'):
                category = f"Strategy: {category.replace('strategy_', '')}"
            
            categories.append(category)
            values.append(value)
        
        # Sort by value
        sorted_indices = np.argsort(values)[::-1]  # Descending order
        categories = [categories[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create bar chart
        bars = ax.bar(categories, values)
        
        # Add reference lines for limits
        ax.axhline(y=self.max_sector_risk, color='r', linestyle='--', label=f'Sector Limit ({self.max_sector_risk}%)')
        ax.axhline(y=self.max_strategy_risk, color='g', linestyle='--', label=f'Strategy Limit ({self.max_strategy_risk}%)')
        
        # Add labels and title
        ax.set_xlabel('Category')
        ax.set_ylabel('Risk Concentration (%)')
        ax.set_title('Portfolio Risk Concentration')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', rotation=0)
        
        # Add legend
        ax.legend()
        
        # Adjust layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def _generate_correlation_heatmap(self):
        """
        Generate correlation heatmap for portfolio positions.
        
        Returns:
        --------
        matplotlib.figure.Figure or None
            Chart figure or None if not enough data
        """
        # Need at least 2 positions with correlation data
        positions_with_corr = [p for p in self.positions if 'correlation' in p and isinstance(p['correlation'], dict)]
        if len(positions_with_corr) < 2:
            return None
        
        # Extract symbols
        symbols = [p.get('symbol', f"Position {i+1}") for i, p in enumerate(positions_with_corr)]
        
        # Create correlation matrix
        n = len(symbols)
        corr_matrix = np.ones((n, n))
        
        # Fill correlation matrix
        for i, pos1 in enumerate(positions_with_corr):
            for j, pos2 in enumerate(positions_with_corr):
                if i == j:
                    continue
                
                symbol1 = pos1.get('symbol', '')
                symbol2 = pos2.get('symbol', '')
                
                # Try to get correlation
                if symbol2 in pos1.get('correlation', {}):
                    corr_matrix[i, j] = pos1['correlation'][symbol2]
                elif symbol1 in pos2.get('correlation', {}):
                    corr_matrix[i, j] = pos2['correlation'][symbol1]
                else:
                    # Default to 0 if correlation not available
                    corr_matrix[i, j] = 0
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
        
        # Add labels and title
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(symbols)
        ax.set_yticklabels(symbols)
        ax.set_title('Position Correlation Heatmap')
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add correlation values in cells
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                              ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig


class RiskManagementIntegrator:
    """
    Integrates risk management components with the options trading system.
    """
    def __init__(self, account_size=100000, risk_profile='moderate'):
        """
        Initialize the risk management integrator.
        
        Parameters:
        -----------
        account_size : float
            Total account size in dollars
        risk_profile : str
            Risk profile ('conservative', 'moderate', or 'aggressive')
        """
        self.account_size = account_size
        self.risk_profile = risk_profile
        
        # Initialize risk management components
        self._initialize_components()
        
        # Initialize portfolio
        self.portfolio = []
    
    def _initialize_components(self):
        """
        Initialize risk management components based on risk profile.
        """
        # Set risk parameters based on profile
        if self.risk_profile == 'conservative':
            max_risk_pct = 1.0
            max_position_pct = 3.0
            max_portfolio_risk = 10.0
            risk_reward_ratio = 3.0
        elif self.risk_profile == 'aggressive':
            max_risk_pct = 3.0
            max_position_pct = 10.0
            max_portfolio_risk = 25.0
            risk_reward_ratio = 1.5
        else:  # moderate
            max_risk_pct = 2.0
            max_position_pct = 5.0
            max_portfolio_risk = 15.0
            risk_reward_ratio = 2.0
        
        # Initialize components
        self.position_sizer = PositionSizer(
            account_size=self.account_size,
            max_risk_pct=max_risk_pct,
            max_position_pct=max_position_pct,
            volatility_adjustment=True,
            kelly_criterion=True
        )
        
        self.exit_calculator = StopLossTakeProfitCalculator(
            risk_reward_ratio=risk_reward_ratio,
            use_volatility=True,
            use_support_resistance=True,
            time_based_exits=True
        )
        
        self.portfolio_manager = PortfolioRiskManager(
            max_portfolio_risk=max_portfolio_risk,
            max_sector_risk=30.0,
            max_strategy_risk=40.0,
            correlation_threshold=0.7
        )
    
    def process_recommendation(self, recommendation, market_data=None):
        """
        Process a trading recommendation and add risk management components.
        
        Parameters:
        -----------
        recommendation : dict
            Trading recommendation from the recommendation engine
        market_data : dict, optional
            Additional market data for risk calculations
            
        Returns:
        --------
        dict
            Enhanced recommendation with risk management details
        """
        # Make a copy of the recommendation
        enhanced_rec = recommendation.copy()
        
        # Extract option data
        option_data = self._extract_option_data(recommendation)
        
        # Calculate position size
        confidence_score = recommendation.get('confidence', {}).get('score', 0.5)
        position_size = self.position_sizer.calculate_position_size(
            option_data, 
            confidence_score=confidence_score
        )
        
        # Calculate exit points
        exit_points = self.exit_calculator.calculate_exit_points(
            option_data,
            support_resistance_levels=market_data.get('support_resistance') if market_data else None
        )
        
        # Add risk management details to recommendation
        enhanced_rec['risk_management'] = {
            'position_sizing': position_size,
            'exit_points': exit_points,
            'risk_profile': self.risk_profile
        }
        
        # Calculate risk metrics
        risk_amount = position_size.get('total_risk', 0)
        exposure = position_size.get('total_cost', 0)
        
        # Create portfolio position
        position = {
            'symbol': recommendation.get('symbol', ''),
            'option_type': option_data.get('putCall', 'CALL'),
            'strike': option_data.get('strike', 0),
            'expiration': option_data.get('expirationDate', ''),
            'contracts': position_size.get('recommended_contracts', 0),
            'exposure': exposure,
            'risk': risk_amount,
            'sector': market_data.get('sector', 'Unknown') if market_data else 'Unknown',
            'strategy': recommendation.get('strategy', 'Unknown'),
            'expected_return': recommendation.get('potential_return', 0),
            'confidence': confidence_score,
            'stop_loss': exit_points.get('final_stop_loss', 0),
            'take_profit': exit_points.get('final_take_profit', 0)
        }
        
        # Add correlation data if available
        if market_data and 'correlations' in market_data:
            position['correlation'] = market_data['correlations']
        
        # Add position to portfolio manager
        position_id = self.portfolio_manager.add_position(position)
        
        # Calculate portfolio risk metrics
        portfolio_risk = self.portfolio_manager.calculate_portfolio_risk()
        
        # Add portfolio context to recommendation
        enhanced_rec['portfolio_context'] = {
            'position_id': position_id,
            'total_positions': portfolio_risk.get('total_positions', 0),
            'total_exposure': portfolio_risk.get('total_exposure', 0),
            'total_risk': portfolio_risk.get('total_risk', 0),
            'diversification_score': portfolio_risk.get('diversification_score', 0),
            'risk_warnings': portfolio_risk.get('risk_warnings', [])
        }
        
        return enhanced_rec
    
    def _extract_option_data(self, recommendation):
        """
        Extract option data from recommendation.
        
        Parameters:
        -----------
        recommendation : dict
            Trading recommendation
            
        Returns:
        --------
        dict
            Option data
        """
        # Initialize with empty dict
        option_data = {}
        
        # Try to extract from different possible locations
        if 'option_data' in recommendation:
            option_data = recommendation['option_data']
        elif 'option' in recommendation:
            option_data = recommendation['option']
        else:
            # Try to build from recommendation fields
            for field in ['symbol', 'strike', 'putCall', 'expirationDate', 'bid', 'ask', 
                         'last', 'volume', 'openInterest', 'impliedVolatility', 'delta',
                         'gamma', 'theta', 'vega', 'rho', 'underlyingPrice']:
                if field in recommendation:
                    option_data[field] = recommendation[field]
        
        return option_data
    
    def generate_portfolio_report(self):
        """
        Generate a comprehensive portfolio risk report.
        
        Returns:
        --------
        dict
            Portfolio risk report
        """
        return self.portfolio_manager.generate_risk_report(account_size=self.account_size)
    
    def get_position_sizing_recommendation(self, option_data, confidence_score=None):
        """
        Get position sizing recommendation for an option.
        
        Parameters:
        -----------
        option_data : dict
            Option data
        confidence_score : float, optional
            Confidence score (0-1)
            
        Returns:
        --------
        dict
            Position sizing recommendation
        """
        return self.position_sizer.calculate_position_size(option_data, confidence_score=confidence_score)
    
    def get_exit_points_recommendation(self, option_data, support_resistance_levels=None):
        """
        Get exit points recommendation for an option.
        
        Parameters:
        -----------
        option_data : dict
            Option data
        support_resistance_levels : dict, optional
            Support and resistance levels
            
        Returns:
        --------
        dict
            Exit points recommendation
        """
        return self.exit_calculator.calculate_exit_points(option_data, support_resistance_levels=support_resistance_levels)
    
    def update_account_size(self, account_size):
        """
        Update account size.
        
        Parameters:
        -----------
        account_size : float
            New account size
        """
        self.account_size = account_size
        self.position_sizer.account_size = account_size
    
    def update_risk_profile(self, risk_profile):
        """
        Update risk profile.
        
        Parameters:
        -----------
        risk_profile : str
            New risk profile ('conservative', 'moderate', or 'aggressive')
        """
        self.risk_profile = risk_profile
        self._initialize_components()
    
    def clear_portfolio(self):
        """
        Clear the portfolio.
        """
        self.portfolio_manager.positions = []
