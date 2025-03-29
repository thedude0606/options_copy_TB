"""
Technical indicators module for options recommendation platform.
Implements various technical indicators for market analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats

class TechnicalIndicators:
    """
    Class to calculate technical indicators for market analysis
    """
    
    def __init__(self, data=None):
        """
        Initialize the technical indicators calculator
        
        Args:
            data (pd.DataFrame): Historical price data
        """
        self.data = data
    
    def calculate_all_indicators(self, price_data):
        """
        Calculate all technical indicators for the given price data
        
        Args:
            price_data (pd.DataFrame): Historical price data with OHLCV columns
            
        Returns:
            pd.DataFrame: DataFrame with all calculated indicators
        """
        if price_data is None or price_data.empty:
            return pd.DataFrame()
            
        # Store original data
        original_data = self.data
        
        # Set instance data to the provided price data
        self.data = price_data
        
        # Initialize results dictionary
        indicators = {}
        
        # Calculate moving averages
        try:
            indicators['sma_20'] = self.sma(period=20, price_col='close')
            indicators['sma_50'] = self.sma(period=50, price_col='close')
            indicators['sma_200'] = self.sma(period=200, price_col='close')
            indicators['ema_12'] = self.ema(period=12, price_col='close')
            indicators['ema_26'] = self.ema(period=26, price_col='close')
        except Exception as e:
            print(f"Error calculating moving averages: {str(e)}")
        
        # Calculate Bollinger Bands
        try:
            bb = self.bollinger_bands(period=20, std_dev=2, price_col='close')
            indicators['bb_middle'] = bb['middle_band']
            indicators['bb_upper'] = bb['upper_band']
            indicators['bb_lower'] = bb['lower_band']
            indicators['bb_width'] = bb['width']
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {str(e)}")
        
        # Calculate RSI
        try:
            indicators['rsi_14'] = self.rsi(period=14, price_col='close')
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
        
        # Calculate MACD
        try:
            macd_data = self.macd(fast_period=12, slow_period=26, signal_period=9, price_col='close')
            indicators['macd_line'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_histogram'] = macd_data['histogram']
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
        
        # Calculate Stochastic
        try:
            if 'high' in price_data.columns and 'low' in price_data.columns:
                stoch = self.stochastic(high_col='high', low_col='low', close_col='close')
                indicators['stoch_k'] = stoch['k']
                indicators['stoch_d'] = stoch['d']
        except Exception as e:
            print(f"Error calculating Stochastic: {str(e)}")
        
        # Calculate ATR
        try:
            if 'high' in price_data.columns and 'low' in price_data.columns:
                indicators['atr_14'] = self.atr(high_col='high', low_col='low', close_col='close', period=14)
        except Exception as e:
            print(f"Error calculating ATR: {str(e)}")
        
        # Calculate ADX
        try:
            if 'high' in price_data.columns and 'low' in price_data.columns:
                adx_data = self.adx(high_col='high', low_col='low', close_col='close', period=14)
                indicators['adx'] = adx_data['adx']
                indicators['di_plus'] = adx_data['di_plus']
                indicators['di_minus'] = adx_data['di_minus']
        except Exception as e:
            print(f"Error calculating ADX: {str(e)}")
        
        # Calculate OBV
        try:
            if 'volume' in price_data.columns:
                indicators['obv'] = self.obv(close_col='close', volume_col='volume')
        except Exception as e:
            print(f"Error calculating OBV: {str(e)}")
        
        # Calculate CMF
        try:
            if 'high' in price_data.columns and 'low' in price_data.columns and 'volume' in price_data.columns:
                indicators['cmf_20'] = self.cmf(high_col='high', low_col='low', close_col='close', volume_col='volume', period=20)
        except Exception as e:
            print(f"Error calculating CMF: {str(e)}")
        
        # Calculate MFI
        try:
            if 'high' in price_data.columns and 'low' in price_data.columns and 'volume' in price_data.columns:
                indicators['mfi_14'] = self.mfi(high_col='high', low_col='low', close_col='close', volume_col='volume', period=14)
        except Exception as e:
            print(f"Error calculating MFI: {str(e)}")
        
        # Calculate CCI
        try:
            if 'high' in price_data.columns and 'low' in price_data.columns:
                indicators['cci_20'] = self.cci(high_col='high', low_col='low', close_col='close', period=20)
        except Exception as e:
            print(f"Error calculating CCI: {str(e)}")
        
        # Create DataFrame with all indicators
        result = pd.DataFrame(indicators, index=price_data.index if hasattr(price_data, 'index') else None)
        
        # Restore original data
        self.data = original_data
        
        return result
    
    def bollinger_bands(self, period=20, std_dev=2, price_col='close'):
        """
        Calculate Bollinger Bands using instance data
        
        Args:
            period (int): Period for moving average
            std_dev (float): Number of standard deviations
            price_col (str): Column name for price data
            
        Returns:
            pd.DataFrame: DataFrame with middle, upper, and lower bands
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        # Calculate middle band (SMA)
        middle_band = self.data[price_col].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = self.data[price_col].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        # Calculate bandwidth
        bandwidth = (upper_band - lower_band) / middle_band
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'width': bandwidth
        }, index=self.data.index if hasattr(self.data, 'index') else None)
        
        return result
    
    def sma(self, period=20, price_col='close'):
        """
        Calculate Simple Moving Average using instance data
        
        Args:
            period (int): Period for SMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: SMA values
        """
        if self.data is None or self.data.empty or len(self.data) < period:
            return pd.Series()
            
        return self.data[price_col].rolling(window=period).mean()
    
    def ema(self, period=20, price_col='close'):
        """
        Calculate Exponential Moving Average using instance data
        
        Args:
            period (int): Period for EMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: EMA values
        """
        if self.data is None or self.data.empty or len(self.data) < period:
            return pd.Series()
            
        return self.data[price_col].ewm(span=period, adjust=False).mean()
    
    def rsi(self, period=14, price_col='close'):
        """
        Calculate Relative Strength Index using instance data
        
        Args:
            period (int): Period for RSI calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: RSI values
        """
        if self.data is None:
            return pd.Series()
        return self.calculate_rsi(self.data, period, price_col)
        
    def calculate_rsi(self, data, period=14, price_col='close'):
        """
        Calculate Relative Strength Index for the given data
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            period (int): Period for RSI calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: RSI values
        """
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series()
            
        # Calculate price changes
        delta = data[price_col].diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses over the period
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate Moving Average Convergence Divergence using instance data
        
        Args:
            fast_period (int): Period for fast EMA
            slow_period (int): Period for slow EMA
            signal_period (int): Period for signal line
            price_col (str): Column name for price data
            
        Returns:
            pd.DataFrame: DataFrame with MACD, signal, and histogram
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        macd_line, signal_line, histogram = self.calculate_macd(
            self.data, 
            fast_period=fast_period, 
            slow_period=slow_period, 
            signal_period=signal_period, 
            price_col=price_col
        )
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }, index=self.data.index if hasattr(self.data, 'index') else None)
        
        return result
    
    def money_flow_index(self, period=14):
        """
        Calculate Money Flow Index using instance data
        
        Args:
            period (int): Period for MFI calculation
            
        Returns:
            pd.Series: MFI values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_mfi(self.data, period=period)
    
    # New indicator methods for Phase 1 enhancements
