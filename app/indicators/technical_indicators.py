"""
Technical indicators module for options recommendation platform.
Implements various technical indicators for market analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats

# Add standalone functions for direct import compatibility
def calculate_rsi(data, period=14, price_col='close', overbought=70, oversold=30):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data (pd.DataFrame): Historical price data
        period (int): Period for RSI calculation
        price_col (str): Column name for price data
        overbought (float): Overbought threshold
        oversold (float): Oversold threshold
        
    Returns:
        pd.DataFrame: DataFrame with RSI values and signals
    """
    rsi_values = TechnicalIndicators.calculate_rsi(data, period, price_col)
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    result['rsi'] = rsi_values
    result['overbought'] = overbought
    result['oversold'] = oversold
    result['signal'] = 'neutral'
    
    # Set signals
    result.loc[result['rsi'] > overbought, 'signal'] = 'bearish'
    result.loc[result['rsi'] < oversold, 'signal'] = 'bullish'
    
    return result

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        data (pd.DataFrame): Historical price data
        fast_period (int): Period for fast EMA
        slow_period (int): Period for slow EMA
        signal_period (int): Period for signal line
        price_col (str): Column name for price data
        
    Returns:
        pd.DataFrame: DataFrame with MACD, signal, and histogram
    """
    macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(
        data, 
        fast_period=fast_period, 
        slow_period=slow_period, 
        signal_period=signal_period, 
        price_col=price_col
    )
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_hist'] = histogram
    result['signal'] = 'neutral'
    
    # Set signals
    result['signal'] = np.where(macd_line > signal_line, 'bullish', 'bearish')
    
    return result

def calculate_bollinger_bands(data, period=20, std_dev=2.0, price_col='close'):
    """
    Calculate Bollinger Bands
    
    Args:
        data (pd.DataFrame): Historical price data
        period (int): Period for moving average
        std_dev (float): Number of standard deviations
        price_col (str): Column name for price data
        
    Returns:
        pd.DataFrame: DataFrame with middle, upper, and lower bands
    """
    middle_band, upper_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(
        data, 
        period=period, 
        std_dev=std_dev, 
        price_col=price_col
    )
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    result['bb_middle'] = middle_band
    result['bb_upper'] = upper_band
    result['bb_lower'] = lower_band
    result['signal'] = 'neutral'
    
    # Set signals
    result.loc[data[price_col] > upper_band, 'signal'] = 'bearish'
    result.loc[data[price_col] < lower_band, 'signal'] = 'bullish'
    
    # Calculate %B (position within bands)
    result['pct_b'] = (data[price_col] - lower_band) / (upper_band - lower_band)
    
    return result

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
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band
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
        return self.calculate_rsi(self.data, period, price_col)
    
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
    
    @staticmethod
    def calculate_rsi(data, period=14, price_col='close'):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data (pd.DataFrame): Historical price data
            period (int): Period for RSI calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: RSI values
        """
        if data.empty or len(data) < period + 1:
            return pd.Series()
        
        # Calculate price changes
        delta = data[price_col].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            data (pd.DataFrame): Historical price data
            fast_period (int): Period for fast EMA
            slow_period (int): Period for slow EMA
            signal_period (int): Period for signal line
            price_col (str): Column name for price data
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        if data.empty or len(data) < slow_period + signal_period:
            return pd.Series(), pd.Series(), pd.Series()
        
        # Calculate EMAs
        ema_fast = data[price_col].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_sma(data, period=20, price_col='close'):
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            data (pd.DataFrame): Historical price data
            period (int): Period for SMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: SMA values
        """
        if data.empty or len(data) < period:
            return pd.Series()
        
        return data[price_col].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data, period=20, price_col='close'):
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            data (pd.DataFrame): Historical price data
            period (int): Period for EMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: EMA values
        """
        if data.empty or len(data) < period:
            return pd.Series()
        
        return data[price_col].ewm(span=period, adjust=False).mean()
        
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2, price_col='close'):
        """
        Calculate Bollinger Bands
        
        Args:
            data (pd.DataFrame): Historical price data
            period (int): Period for moving average
            std_dev (float): Number of standard deviations
            price_col (str): Column name for price data
            
        Returns:
            tuple: (Middle Band, Upper Band, Lower Band)
        """
        if data.empty or len(data) < period:
            return pd.Series(), pd.Series(), pd.Series()
        
        # Calculate middle band (SMA)
        middle_band = data[price_col].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = data[price_col].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return middle_band, upper_band, lower_band
