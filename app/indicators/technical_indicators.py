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
