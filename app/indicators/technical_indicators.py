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
            
        return self.calculate_bollinger_bands(
            self.data,
            period=period,
            std_dev=std_dev,
            price_col=price_col
        )
            
    def calculate_bollinger_bands(self, data, period=20, std_dev=2, price_col='close'):
        """
        Calculate Bollinger Bands for the given data
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            period (int): Period for moving average
            std_dev (float): Number of standard deviations
            price_col (str): Column name for price data
            
        Returns:
            pd.DataFrame: DataFrame with middle, upper, and lower bands
        """
        if data is None or data.empty or len(data) < period:
            return pd.DataFrame()
            
        # Calculate middle band (SMA)
        middle_band = data[price_col].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = data[price_col].rolling(window=period).std()
        
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
        }, index=data.index if hasattr(data, 'index') else None)
        
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
        
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate Moving Average Convergence Divergence for the given data
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            fast_period (int): Period for fast EMA
            slow_period (int): Period for slow EMA
            signal_period (int): Period for signal line
            price_col (str): Column name for price data
            
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        if data is None or data.empty or len(data) < slow_period:
            return pd.Series(), pd.Series(), pd.Series()
            
        # Calculate fast and slow EMAs
        fast_ema = data[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram (MACD line - signal line)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
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
    
    def stochastic(self, period=14, k_smoothing=3, d_smoothing=3, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Stochastic Oscillator using instance data
        
        Args:
            period (int): Period for Stochastic calculation
            k_smoothing (int): Smoothing period for %K
            d_smoothing (int): Smoothing period for %D
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            
        Returns:
            pd.DataFrame: DataFrame with %K and %D values
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        k, d = self.calculate_stochastic(
            self.data,
            period=period,
            k_smoothing=k_smoothing,
            d_smoothing=d_smoothing,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col
        )
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'k': k,
            'd': d
        }, index=self.data.index if hasattr(self.data, 'index') else None)
        
        return result
        
    def calculate_stochastic(self, data, period=14, k_smoothing=3, d_smoothing=3, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Stochastic Oscillator for the given data
        
        Args:
            data (pd.DataFrame): Price data with high, low, and close columns
            period (int): Period for Stochastic calculation
            k_smoothing (int): Smoothing period for %K
            d_smoothing (int): Smoothing period for %D
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            
        Returns:
            tuple: (%K, %D)
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series(), pd.Series()
            
        # Calculate highest high and lowest low over the period
        highest_high = data[high_col].rolling(window=period).max()
        lowest_low = data[low_col].rolling(window=period).min()
        
        # Calculate raw %K
        raw_k = 100 * ((data[close_col] - lowest_low) / (highest_high - lowest_low))
        
        # Apply smoothing to %K if specified
        if k_smoothing > 1:
            k = raw_k.rolling(window=k_smoothing).mean()
        else:
            k = raw_k
        
        # Calculate %D (moving average of %K)
        d = k.rolling(window=d_smoothing).mean()
        
        return k, d
        
    def atr(self, period=14, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Average True Range using instance data
        
        Args:
            period (int): Period for ATR calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            
        Returns:
            pd.Series: ATR values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_atr(
            self.data,
            period=period,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col
        )
        
    def calculate_atr(self, data, period=14, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Average True Range for the given data
        
        Args:
            data (pd.DataFrame): Price data with high, low, and close columns
            period (int): Period for ATR calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            
        Returns:
            pd.Series: ATR values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate true range
        high_low = data[high_col] - data[low_col]
        high_close = (data[high_col] - data[close_col].shift()).abs()
        low_close = (data[low_col] - data[close_col].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR (simple moving average of true range)
        atr = true_range.rolling(window=period).mean()
        
        return atr
        
    def adx(self, period=14, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Average Directional Index using instance data
        
        Args:
            period (int): Period for ADX calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            
        Returns:
            pd.DataFrame: DataFrame with ADX, +DI, and -DI values
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        adx, di_plus, di_minus = self.calculate_adx(
            self.data,
            period=period,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col
        )
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }, index=self.data.index if hasattr(self.data, 'index') else None)
        
        return result
        
    def calculate_adx(self, data, period=14, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Average Directional Index for the given data
        
        Args:
            data (pd.DataFrame): Price data with high, low, and close columns
            period (int): Period for ADX calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            
        Returns:
            tuple: (ADX, +DI, -DI)
        """
        if data is None or data.empty or len(data) < period * 2:
            return pd.Series(), pd.Series(), pd.Series()
            
        # Calculate price changes
        high_change = data[high_col].diff()
        low_change = data[low_col].diff()
        
        # Calculate +DM and -DM
        plus_dm = pd.Series(0, index=data.index)
        minus_dm = pd.Series(0, index=data.index)
        
        # Set conditions for +DM and -DM
        condition1 = (high_change > 0) & (high_change > low_change.abs())
        condition2 = (low_change < 0) & (low_change.abs() > high_change)
        
        plus_dm[condition1] = high_change[condition1]
        minus_dm[condition2] = low_change.abs()[condition2]
        
        # Calculate ATR
        atr = self.calculate_atr(data, period, high_col, low_col, close_col)
        
        # Calculate smoothed +DM and -DM
        plus_dm_smoothed = plus_dm.rolling(window=period).sum()
        minus_dm_smoothed = minus_dm.rolling(window=period).sum()
        
        # Calculate +DI and -DI
        di_plus = 100 * (plus_dm_smoothed / atr)
        di_minus = 100 * (minus_dm_smoothed / atr)
        
        # Calculate DX
        dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
        
        # Calculate ADX (smoothed DX)
        adx = dx.rolling(window=period).mean()
        
        return adx, di_plus, di_minus
        
    def obv(self, close_col='close', volume_col='volume'):
        """
        Calculate On-Balance Volume using instance data
        
        Args:
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: OBV values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_obv(
            self.data,
            close_col=close_col,
            volume_col=volume_col
        )
        
    def calculate_obv(self, data, close_col='close', volume_col='volume'):
        """
        Calculate On-Balance Volume for the given data
        
        Args:
            data (pd.DataFrame): Price data with close and volume columns
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: OBV values
        """
        if data is None or data.empty or len(data) < 2:
            return pd.Series()
            
        # Calculate price changes
        price_change = data[close_col].diff()
        
        # Create OBV series
        obv = pd.Series(0, index=data.index)
        
        # Set initial OBV value
        obv.iloc[0] = 0
        
        # Calculate OBV based on price change direction
        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:  # Price up
                obv.iloc[i] = obv.iloc[i-1] + data[volume_col].iloc[i]
            elif price_change.iloc[i] < 0:  # Price down
                obv.iloc[i] = obv.iloc[i-1] - data[volume_col].iloc[i]
            else:  # Price unchanged
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
        
    def cmf(self, period=20, high_col='high', low_col='low', close_col='close', volume_col='volume'):
        """
        Calculate Chaikin Money Flow using instance data
        
        Args:
            period (int): Period for CMF calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: CMF values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_cmf(
            self.data,
            period=period,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col
        )
        
    def calculate_cmf(self, data, period=20, high_col='high', low_col='low', close_col='close', volume_col='volume'):
        """
        Calculate Chaikin Money Flow for the given data
        
        Args:
            data (pd.DataFrame): Price data with high, low, close, and volume columns
            period (int): Period for CMF calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: CMF values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate Money Flow Multiplier
        high_low_range = data[high_col] - data[low_col]
        money_flow_multiplier = ((data[close_col] - data[low_col]) - (data[high_col] - data[close_col])) / high_low_range
        money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * data[volume_col]
        
        # Calculate Chaikin Money Flow
        cmf = money_flow_volume.rolling(window=period).sum() / data[volume_col].rolling(window=period).sum()
        
        return cmf
        
    def mfi(self, period=14, high_col='high', low_col='low', close_col='close', volume_col='volume'):
        """
        Calculate Money Flow Index using instance data
        
        Args:
            period (int): Period for MFI calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: MFI values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_mfi(
            self.data,
            period=period,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col
        )
        
    def calculate_mfi(self, data, period=14, high_col='high', low_col='low', close_col='close', volume_col='volume'):
        """
        Calculate Money Flow Index for the given data
        
        Args:
            data (pd.DataFrame): Price data with high, low, close, and volume columns
            period (int): Period for MFI calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: MFI values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate typical price
        typical_price = (data[high_col] + data[low_col] + data[close_col]) / 3
        
        # Calculate money flow
        money_flow = typical_price * data[volume_col]
        
        # Calculate positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = pd.Series(0, index=data.index)
        negative_flow = pd.Series(0, index=data.index)
        
        positive_flow[price_change > 0] = money_flow[price_change > 0]
        negative_flow[price_change < 0] = money_flow[price_change < 0]
        
        # Calculate positive and negative money flow sums over the period
        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()
        
        # Calculate money flow ratio
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
        
    def cci(self, period=20, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Commodity Channel Index using instance data
        
        Args:
            period (int): Period for CCI calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            
        Returns:
            pd.Series: CCI values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_cci(
            self.data,
            period=period,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col
        )
        
    def calculate_cci(self, data, period=20, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Commodity Channel Index for the given data
        
        Args:
            data (pd.DataFrame): Price data with high, low, and close columns
            period (int): Period for CCI calculation
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            
        Returns:
            pd.Series: CCI values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate typical price
        typical_price = (data[high_col] + data[low_col] + data[close_col]) / 3
        
        # Calculate simple moving average of typical price
        tp_sma = typical_price.rolling(window=period).mean()
        
        # Calculate mean deviation
        mean_deviation = pd.Series(0, index=data.index)
        for i in range(period - 1, len(data)):
            mean_deviation.iloc[i] = abs(typical_price.iloc[i-period+1:i+1] - tp_sma.iloc[i]).mean()
        
        # Calculate CCI
        cci = (typical_price - tp_sma) / (0.015 * mean_deviation)
        
        return cci
    
    def imi(self, period=14, open_col='open', close_col='close'):
        """
        Calculate Intraday Momentum Index using instance data
        
        Args:
            period (int): Period for IMI calculation
            open_col (str): Column name for open price data
            close_col (str): Column name for close price data
            
        Returns:
            pd.Series: IMI values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_imi(
            self.data,
            period=period,
            open_col=open_col,
            close_col=close_col
        )
        
    def calculate_imi(self, data, period=14, open_col='open', close_col='close'):
        """
        Calculate Intraday Momentum Index for the given data
        
        Args:
            data (pd.DataFrame): Price data with open and close columns
            period (int): Period for IMI calculation
            open_col (str): Column name for open price data
            close_col (str): Column name for close price data
            
        Returns:
            pd.Series: IMI values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate intraday price changes (close - open)
        intraday_change = data[close_col] - data[open_col]
        
        # Separate gains (up days) and losses (down days)
        gains = intraday_change.copy()
        losses = intraday_change.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate sum of gains and losses over the period
        gains_sum = gains.rolling(window=period).sum()
        losses_sum = losses.rolling(window=period).sum()
        
        # Calculate IMI
        imi = 100 * (gains_sum / (gains_sum + losses_sum))
        
        return imi
    
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
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss over the period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # New indicator methods for Phase 1 enhancements

    def calculate_fair_value_gap(self, data, high_col='high', low_col='low', close_col='close', threshold=0.5):
        """
        Calculate Fair Value Gap (FVG) for the given data.
        A Fair Value Gap occurs when price moves rapidly in one direction, leaving an imbalance or "gap" in price.
        
        Args:
            data (pd.DataFrame): Price data with high, low, and close columns
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            threshold (float): Minimum percentage gap to be considered a FVG
            
        Returns:
            pd.Series: Fair Value Gap values (positive for bullish gaps, negative for bearish gaps)
        """
        if data is None or data.empty or len(data) < 3:
            return pd.Series(dtype=float)
            
        # Initialize result series
        fvg = pd.Series(0.0, index=data.index)
        
        # Calculate FVG for each candle (starting from the 3rd candle)
        for i in range(2, len(data)):
            # Get the three consecutive candles
            candle1 = data.iloc[i-2]
            candle2 = data.iloc[i-1]
            candle3 = data.iloc[i]
            
            # Bullish FVG: candle1 high < candle3 low
            if candle1[high_col] < candle3[low_col]:
                gap_size = candle3[low_col] - candle1[high_col]
                avg_price = (candle1[close_col] + candle2[close_col] + candle3[close_col]) / 3
                gap_percent = gap_size / avg_price * 100
                
                if gap_percent > threshold:
                    fvg.iloc[i] = gap_percent
            
            # Bearish FVG: candle1 low > candle3 high
            elif candle1[low_col] > candle3[high_col]:
                gap_size = candle1[low_col] - candle3[high_col]
                avg_price = (candle1[close_col] + candle2[close_col] + candle3[close_col]) / 3
                gap_percent = gap_size / avg_price * 100
                
                if gap_percent > threshold:
                    fvg.iloc[i] = -gap_percent  # Negative for bearish gaps
        
        return fvg

    def calculate_liquidity_zones(self, data, high_col='high', low_col='low', volume_col='volume', lookback=20, threshold=1.5):
        """
        Calculate Liquidity Zones (areas of high volume/liquidity) for the given data.
        Liquidity zones are price levels where significant trading activity has occurred,
        often acting as support or resistance levels.
        
        Args:
            data (pd.DataFrame): Price data with high, low, and volume columns
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            volume_col (str): Column name for volume data
            lookback (int): Number of periods to look back for identifying liquidity zones
            threshold (float): Volume threshold multiplier to identify significant liquidity zones
            
        Returns:
            pd.DataFrame: DataFrame with support and resistance liquidity zones
        """
        if data is None or data.empty or len(data) < lookback:
            return pd.DataFrame(columns=['price', 'strength', 'type'])
            
        # Initialize result
        liquidity_zones = []
        
        # Calculate average volume
        avg_volume = data[volume_col].rolling(window=lookback).mean()
        
        # Identify high volume bars
        high_volume_bars = data[volume_col] > (avg_volume * threshold)
        
        # Process the data to find liquidity zones
        for i in range(lookback, len(data)):
            if high_volume_bars.iloc[i]:
                # Get the price range for this bar
                high_price = data[high_col].iloc[i]
                low_price = data[low_col].iloc[i]
                volume = data[volume_col].iloc[i]
                
                # Calculate the strength based on volume relative to average
                strength = volume / avg_volume.iloc[i]
                
                # Add support zone (bottom of high volume bar)
                liquidity_zones.append({
                    'price': low_price,
                    'strength': strength,
                    'type': 'support'
                })
                
                # Add resistance zone (top of high volume bar)
                liquidity_zones.append({
                    'price': high_price,
                    'strength': strength,
                    'type': 'resistance'
                })
        
        # Convert to DataFrame
        if liquidity_zones:
            result = pd.DataFrame(liquidity_zones)
            
            # Group by similar price levels (within 0.5% range) and sum the strength
            result['price_group'] = result['price'].apply(lambda x: round(x, 2))
            grouped = result.groupby(['price_group', 'type']).agg({'strength': 'sum'}).reset_index()
            
            # Sort by strength in descending order
            grouped = grouped.sort_values('strength', ascending=False)
            
            # Keep only the top 5 support and top 5 resistance zones
            supports = grouped[grouped['type'] == 'support'].head(5)
            resistances = grouped[grouped['type'] == 'resistance'].head(5)
            
            # Combine and return
            return pd.concat([supports, resistances]).drop('price_group', axis=1)
        else:
            return pd.DataFrame(columns=['price', 'strength', 'type'])
            
    def calculate_moving_averages(self, data, close_col='close', periods=[9, 21, 50, 200]):
        """
        Calculate multiple moving averages for the given data.
        
        Args:
            data (pd.DataFrame): Price data with close column
            close_col (str): Column name for close price data
            periods (list): List of periods for moving average calculations
            
        Returns:
            pd.DataFrame: DataFrame with moving averages for each period
        """
        if data is None or data.empty:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        
        # Calculate simple moving averages for each period
        for period in periods:
            result[f'sma_{period}'] = data[close_col].rolling(window=period).mean()
            
        # Calculate exponential moving averages for each period
        for period in periods:
            result[f'ema_{period}'] = data[close_col].ewm(span=period, adjust=False).mean()
            
        # Calculate moving average crossovers
        if len(periods) >= 2:
            # Sort periods to ensure consistent crossover calculations
            sorted_periods = sorted(periods)
            
            for i in range(len(sorted_periods) - 1):
                short_period = sorted_periods[i]
                long_period = sorted_periods[i + 1]
                
                # Calculate crossover signals (1 for bullish crossover, -1 for bearish, 0 for no crossover)
                short_ma = result[f'ema_{short_period}']
                long_ma = result[f'ema_{long_period}']
                
                crossover = pd.Series(0, index=data.index)
                crossover[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1  # Bullish crossover
                crossover[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1  # Bearish crossover
                
                result[f'crossover_{short_period}_{long_period}'] = crossover
        
        return result
        
    def calculate_volatility(self, data, close_col='close', high_col='high', low_col='low', periods=[14, 30, 60]):
        """
        Calculate various volatility indicators for the given data.
        
        Args:
            data (pd.DataFrame): Price data with close, high, and low columns
            close_col (str): Column name for close price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            periods (list): List of periods for volatility calculations
            
        Returns:
            pd.DataFrame: DataFrame with volatility indicators
        """
        if data is None or data.empty:
            return pd.DataFrame()
            
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        
        # Calculate daily returns
        returns = data[close_col].pct_change()
        
        # Calculate historical volatility (standard deviation of returns) for each period
        for period in periods:
            result[f'hist_vol_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        
        # Calculate Average True Range (ATR) for each period
        for period in periods:
            # Calculate True Range
            tr1 = data[high_col] - data[low_col]  # Current high - current low
            tr2 = abs(data[high_col] - data[close_col].shift(1))  # Current high - previous close
            tr3 = abs(data[low_col] - data[close_col].shift(1))  # Current low - previous close
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            result[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        # Calculate Bollinger Bands volatility (bandwidth) for default period (20)
        bb_period = 20
        if len(data) >= bb_period:
            sma = data[close_col].rolling(window=bb_period).mean()
            std_dev = data[close_col].rolling(window=bb_period).std()
            
            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)
            
            # Calculate bandwidth: (upper - lower) / middle
            result['bb_bandwidth'] = (upper_band - lower_band) / sma
        
        # Calculate Chaikin Volatility
        chaikin_period = 10
        if len(data) >= chaikin_period:
            high_low_diff = data[high_col] - data[low_col]
            ema_high_low = high_low_diff.ewm(span=chaikin_period, adjust=False).mean()
            result['chaikin_volatility'] = (ema_high_low / ema_high_low.shift(chaikin_period) - 1) * 100
        
        # Volatility regime classification
        if 'hist_vol_30' in result.columns:
            # Calculate long-term average volatility (using 60-day if available, otherwise 30-day)
            if 'hist_vol_60' in result.columns:
                long_term_vol = result['hist_vol_60'].rolling(window=90).mean()
            else:
                long_term_vol = result['hist_vol_30'].rolling(window=90).mean()
            
            # Classify volatility regime
            regime = pd.Series('normal', index=data.index)
            regime[result['hist_vol_30'] > (1.5 * long_term_vol)] = 'high'
            regime[result['hist_vol_30'] < (0.5 * long_term_vol)] = 'low'
            
            result['volatility_regime'] = regime
        
        return result

    def calculate_sma(self, data, period=20, price_col='close'):
        """
        Calculate Simple Moving Average for the given data.
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            period (int): Period for SMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: SMA values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series(dtype=float)
            
        return data[price_col].rolling(window=period).mean()
        
    def calculate_ema(self, data, period=20, price_col='close'):
        """
        Calculate Exponential Moving Average for the given data.
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            period (int): Period for EMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: EMA values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series(dtype=float)
            
        return data[price_col].ewm(span=period, adjust=False).mean()
        
    def calculate_macd_signal(self, data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate MACD Signal Line for the given data.
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            fast_period (int): Period for fast EMA calculation
            slow_period (int): Period for slow EMA calculation
            signal_period (int): Period for signal line calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: MACD Signal Line values
        """
        if data is None or data.empty or len(data) < max(fast_period, slow_period, signal_period):
            return pd.Series(dtype=float)
            
        # Calculate MACD Line
        fast_ema = data[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[price_col].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        
        # Calculate Signal Line (EMA of MACD Line)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return signal_line
        
    def calculate_macd_histogram(self, data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate MACD Histogram for the given data.
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            fast_period (int): Period for fast EMA calculation
            slow_period (int): Period for slow EMA calculation
            signal_period (int): Period for signal line calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: MACD Histogram values
        """
        if data is None or data.empty or len(data) < max(fast_period, slow_period, signal_period):
            return pd.Series(dtype=float)
            
        # Calculate MACD Line
        fast_ema = data[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[price_col].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        
        # Calculate Signal Line (EMA of MACD Line)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate Histogram (MACD Line - Signal Line)
        histogram = macd_line - signal_line
        
        return histogram
        
    def calculate_cmo(self, data, period=14, price_col='close'):
        """
        Calculate Chande Momentum Oscillator for the given data.
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            period (int): Period for CMO calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: CMO values
        """
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series(dtype=float)
            
        # Calculate price changes
        price_changes = data[price_col].diff()
        
        # Separate gains and losses
        gains = price_changes.copy()
        losses = price_changes.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate sum of gains and losses over the period
        sum_gains = gains.rolling(window=period).sum()
        sum_losses = losses.rolling(window=period).sum()
        
        # Calculate CMO
        cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
        
        return cmo
        
    def calculate_stochastic_rsi(self, data, period=14, smooth_k=3, smooth_d=3, price_col='close'):
        """
        Calculate Stochastic RSI for the given data.
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            period (int): Period for RSI and Stochastic calculation
            smooth_k (int): Smoothing period for %K line
            smooth_d (int): Smoothing period for %D line
            price_col (str): Column name for price data
            
        Returns:
            tuple: (Stochastic RSI %K, Stochastic RSI %D)
        """
        if data is None or data.empty or len(data) < period + smooth_k + smooth_d:
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        # Calculate RSI
        rsi = self.calculate_rsi(data, period=period, price_col=price_col)
        
        # Calculate Stochastic RSI
        stoch_rsi = pd.Series(0.0, index=data.index)
        
        for i in range(period, len(rsi)):
            rsi_window = rsi.iloc[i-period+1:i+1]
            if not rsi_window.isna().all():
                min_rsi = rsi_window.min()
                max_rsi = rsi_window.max()
                
                if max_rsi - min_rsi != 0:
                    stoch_rsi.iloc[i] = (rsi.iloc[i] - min_rsi) / (max_rsi - min_rsi)
        
        # Calculate %K (smoothed Stochastic RSI)
        stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean() * 100
        
        # Calculate %D (smoothed %K)
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()
        
        return stoch_rsi_k, stoch_rsi_d
        
    def calculate_adl(self, data, high_col='high', low_col='low', close_col='close', volume_col='volume'):
        """
        Calculate Accumulation/Distribution Line for the given data.
        
        Args:
            data (pd.DataFrame): Price data with high, low, close, and volume columns
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: A/D Line values
        """
        if data is None or data.empty:
            return pd.Series(dtype=float)
            
        # Calculate Money Flow Multiplier
        high_low_range = data[high_col] - data[low_col]
        
        # Avoid division by zero
        high_low_range = high_low_range.replace(0, np.nan)
        
        close_loc = ((data[close_col] - data[low_col]) - (data[high_col] - data[close_col])) / high_low_range
        close_loc = close_loc.fillna(0)  # Replace NaN with 0
        
        # Calculate Money Flow Volume
        money_flow_volume = close_loc * data[volume_col]
        
        # Calculate A/D Line (cumulative sum of Money Flow Volume)
        ad_line = money_flow_volume.cumsum()
        
        return ad_line
        
    def calculate_adaptive_moving_average(self, data, er_period=10, fast_period=2, slow_period=30, price_col='close'):
        """
        Calculate Kaufman's Adaptive Moving Average (KAMA) for the given data.
        
        Args:
            data (pd.DataFrame): Price data with at least the specified price column
            er_period (int): Period for Efficiency Ratio calculation
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: KAMA values
        """
        if data is None or data.empty or len(data) < er_period + 1:
            return pd.Series(dtype=float)
            
        # Calculate price change and direction
        price_change = abs(data[price_col] - data[price_col].shift(er_period))
        volatility = abs(data[price_col].diff()).rolling(window=er_period).sum()
        
        # Calculate Efficiency Ratio
        er = price_change / volatility
        er = er.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero
        
        # Calculate smoothing constant
        fast_alpha = 2 / (fast_period + 1)
        slow_alpha = 2 / (slow_period + 1)
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # Initialize KAMA
        kama = pd.Series(0.0, index=data.index)
        kama.iloc[er_period] = data[price_col].iloc[er_period]
        
        # Calculate KAMA
        for i in range(er_period + 1, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data[price_col].iloc[i] - kama.iloc[i-1])
        
        return kama
