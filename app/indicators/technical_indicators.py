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
            indicators['obv'] = self.obv(close_col='close', volume_col='volume')
        except Exception as e:
            print(f"Error calculating OBV: {str(e)}")
        
        # Calculate CCI
        try:
            if 'high' in price_data.columns and 'low' in price_data.columns:
                indicators['cci_20'] = self.cci(period=20, high_col='high', low_col='low', close_col='close')
        except Exception as e:
            print(f"Error calculating CCI: {str(e)}")
        
        # Calculate MFI
        try:
            if 'high' in price_data.columns and 'low' in price_data.columns and 'volume' in price_data.columns:
                indicators['mfi_14'] = self.mfi(period=14, high_col='high', low_col='low', close_col='close', volume_col='volume')
        except Exception as e:
            print(f"Error calculating MFI: {str(e)}")
        
        # Calculate IMI
        try:
            if 'open' in price_data.columns:
                indicators['imi_14'] = self.imi(period=14, open_col='open', close_col='close')
        except Exception as e:
            print(f"Error calculating IMI: {str(e)}")
        
        # Restore original data
        self.data = original_data
        
        # Convert indicators dictionary to DataFrame
        return pd.DataFrame(indicators)
    
    def sma(self, period=20, price_col='close'):
        """
        Calculate Simple Moving Average using instance data
        
        Args:
            period (int): Period for SMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: SMA values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_sma(self.data, period=period, price_col=price_col)
    
    def calculate_sma(self, data, period=20, price_col='close'):
        """
        Calculate Simple Moving Average for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            period (int): Period for SMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: SMA values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        return data[price_col].rolling(window=period).mean()
    
    def ema(self, period=20, price_col='close'):
        """
        Calculate Exponential Moving Average using instance data
        
        Args:
            period (int): Period for EMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: EMA values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_ema(self.data, period=period, price_col=price_col)
    
    def calculate_ema(self, data, period=20, price_col='close'):
        """
        Calculate Exponential Moving Average for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            period (int): Period for EMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: EMA values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        return data[price_col].ewm(span=period, adjust=False).mean()
    
    def bollinger_bands(self, period=20, std_dev=2, price_col='close'):
        """
        Calculate Bollinger Bands using instance data
        
        Args:
            period (int): Period for Bollinger Bands calculation
            std_dev (float): Number of standard deviations for bands
            price_col (str): Column name for price data
            
        Returns:
            dict: Dictionary with middle_band, upper_band, lower_band, and width
        """
        if self.data is None or self.data.empty:
            return {'middle_band': pd.Series(), 'upper_band': pd.Series(), 'lower_band': pd.Series(), 'width': pd.Series()}
            
        return self.calculate_bollinger_bands(self.data, period=period, std_dev=std_dev, price_col=price_col)
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2, price_col='close'):
        """
        Calculate Bollinger Bands for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            period (int): Period for Bollinger Bands calculation
            std_dev (float): Number of standard deviations for bands
            price_col (str): Column name for price data
            
        Returns:
            dict: Dictionary with middle_band, upper_band, lower_band, and width
        """
        if data is None or data.empty or len(data) < period:
            return {'middle_band': pd.Series(), 'upper_band': pd.Series(), 'lower_band': pd.Series(), 'width': pd.Series()}
            
        # Calculate middle band (SMA)
        middle_band = data[price_col].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = data[price_col].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Calculate bandwidth
        width = (upper_band - lower_band) / middle_band
        
        # Return as dictionary
        return {
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'width': width
        }
    
    def rsi(self, period=14, price_col='close'):
        """
        Calculate Relative Strength Index using instance data
        
        Args:
            period (int): Period for RSI calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: RSI values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_rsi(self.data, period=period, price_col=price_col)
    
    def calculate_rsi(self, data, period=14, price_col='close'):
        """
        Calculate Relative Strength Index for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            period (int): Period for RSI calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: RSI values
        """
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series()
            
        # Calculate price changes
        delta = data[price_col].diff()
        
        # Create gain and loss series
        gain = delta.copy()
        loss = delta.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
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
            dict: Dictionary with macd, signal, and histogram
        """
        if self.data is None or self.data.empty:
            return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}
            
        return self.calculate_macd(self.data, fast_period=fast_period, slow_period=slow_period, signal_period=signal_period, price_col=price_col)
    
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate Moving Average Convergence Divergence for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            fast_period (int): Period for fast EMA
            slow_period (int): Period for slow EMA
            signal_period (int): Period for signal line
            price_col (str): Column name for price data
            
        Returns:
            dict: Dictionary with macd, signal, and histogram
        """
        if data is None or data.empty or len(data) < slow_period + signal_period:
            return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}
            
        # Calculate fast and slow EMAs
        fast_ema = data[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Return as dictionary
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_macd_signal(self, data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate MACD signal line for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            fast_period (int): Period for fast EMA
            slow_period (int): Period for slow EMA
            signal_period (int): Period for signal line
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: MACD signal line values
        """
        macd_data = self.calculate_macd(data, fast_period, slow_period, signal_period, price_col)
        return macd_data['signal']
    
    def calculate_macd_histogram(self, data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate MACD histogram for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            fast_period (int): Period for fast EMA
            slow_period (int): Period for slow EMA
            signal_period (int): Period for signal line
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: MACD histogram values
        """
        macd_data = self.calculate_macd(data, fast_period, slow_period, signal_period, price_col)
        return macd_data['histogram']
    
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
            dict: Dictionary with k and d values
        """
        if self.data is None or self.data.empty:
            return {'k': pd.Series(), 'd': pd.Series()}
            
        return self.calculate_stochastic(
            self.data,
            period=period,
            k_smoothing=k_smoothing,
            d_smoothing=d_smoothing,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col
        )
    
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
            dict: Dictionary with k and d values
        """
        if data is None or data.empty or len(data) < period:
            return {'k': pd.Series(), 'd': pd.Series()}
            
        # Calculate highest high and lowest low over the period
        highest_high = data[high_col].rolling(window=period).max()
        lowest_low = data[low_col].rolling(window=period).min()
        
        # Calculate %K
        k_raw = 100 * ((data[close_col] - lowest_low) / (highest_high - lowest_low))
        
        # Apply smoothing to %K if specified
        if k_smoothing > 1:
            k = k_raw.rolling(window=k_smoothing).mean()
        else:
            k = k_raw
        
        # Calculate %D (moving average of %K)
        d = k.rolling(window=d_smoothing).mean()
        
        # Return as dictionary
        return {'k': k, 'd': d}
    
    def calculate_stochastic_rsi(self, data, rsi_period=14, stoch_period=14, k_smoothing=3, d_smoothing=3, price_col='close'):
        """
        Calculate Stochastic RSI for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            rsi_period (int): Period for RSI calculation
            stoch_period (int): Period for Stochastic calculation
            k_smoothing (int): Smoothing period for %K
            d_smoothing (int): Smoothing period for %D
            price_col (str): Column name for price data
            
        Returns:
            dict: Dictionary with k and d values
        """
        if data is None or data.empty or len(data) < rsi_period + stoch_period:
            return {'k': pd.Series(), 'd': pd.Series()}
            
        # Calculate RSI
        rsi_values = self.calculate_rsi(data, period=rsi_period, price_col=price_col)
        
        # Create a temporary DataFrame with RSI values
        temp_data = pd.DataFrame({'close': rsi_values, 'high': rsi_values, 'low': rsi_values})
        
        # Calculate Stochastic on RSI values
        return self.calculate_stochastic(
            temp_data,
            period=stoch_period,
            k_smoothing=k_smoothing,
            d_smoothing=d_smoothing,
            high_col='high',
            low_col='low',
            close_col='close'
        )
    
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
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series()
            
        # Calculate true range
        tr1 = data[high_col] - data[low_col]  # Current high - current low
        tr2 = abs(data[high_col] - data[close_col].shift(1))  # Current high - previous close
        tr3 = abs(data[low_col] - data[close_col].shift(1))  # Current low - previous close
        
        # Get the maximum of the three true ranges
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
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
            dict: Dictionary with adx, di_plus, and di_minus values
        """
        if self.data is None or self.data.empty:
            return {'adx': pd.Series(), 'di_plus': pd.Series(), 'di_minus': pd.Series()}
            
        return self.calculate_adx(
            self.data,
            period=period,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col
        )
    
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
            dict: Dictionary with adx, di_plus, and di_minus values
        """
        if data is None or data.empty or len(data) < 2 * period + 1:
            return {'adx': pd.Series(), 'di_plus': pd.Series(), 'di_minus': pd.Series()}
            
        # Calculate True Range
        tr1 = data[high_col] - data[low_col]
        tr2 = abs(data[high_col] - data[close_col].shift(1))
        tr3 = abs(data[low_col] - data[close_col].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = data[high_col] - data[high_col].shift(1)
        down_move = data[low_col].shift(1) - data[low_col]
        
        # Calculate +DM and -DM
        plus_dm = pd.Series(0, index=data.index)
        minus_dm = pd.Series(0, index=data.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        # Calculate smoothed TR, +DM, and -DM
        tr_period = tr.rolling(window=period).sum()
        plus_dm_period = plus_dm.rolling(window=period).sum()
        minus_dm_period = minus_dm.rolling(window=period).sum()
        
        # Calculate +DI and -DI
        di_plus = 100 * (plus_dm_period / tr_period)
        di_minus = 100 * (minus_dm_period / tr_period)
        
        # Calculate DX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        # Return as dictionary
        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }
    
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
            
        return self.calculate_obv(self.data, close_col=close_col, volume_col=volume_col)
    
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
            
        # Initialize OBV with first volume value
        obv = pd.Series(0, index=data.index)
        
        # Calculate OBV
        for i in range(1, len(data)):
            if data[close_col].iloc[i] > data[close_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data[volume_col].iloc[i]
            elif data[close_col].iloc[i] < data[close_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data[volume_col].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_adl(self, data, high_col='high', low_col='low', close_col='close', volume_col='volume'):
        """
        Calculate Accumulation/Distribution Line for the given data
        
        Args:
            data (pd.DataFrame): Price data with high, low, close, and volume columns
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: ADL values
        """
        if data is None or data.empty:
            return pd.Series()
            
        # Calculate Money Flow Multiplier
        mfm = ((data[close_col] - data[low_col]) - (data[high_col] - data[close_col])) / (data[high_col] - data[low_col])
        
        # Handle division by zero
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfm = mfm.fillna(0)
        
        # Calculate Money Flow Volume
        mfv = mfm * data[volume_col]
        
        # Calculate ADL
        adl = mfv.cumsum()
        
        return adl
    
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
        if data is None or data.empty or len(data) < period + 1:
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
            
        # Calculate gains and losses
        gains = pd.Series(0, index=data.index)
        losses = pd.Series(0, index=data.index)
        
        # Determine if close is higher or lower than open
        gains[(data[close_col] > data[open_col])] = data[close_col] - data[open_col]
        losses[(data[close_col] < data[open_col])] = data[open_col] - data[close_col]
        
        # Calculate sum of gains and losses over the period
        gains_sum = gains.rolling(window=period).sum()
        losses_sum = losses.rolling(window=period).sum()
        
        # Calculate IMI
        imi = 100 * (gains_sum / (gains_sum + losses_sum))
        
        return imi
    
    def calculate_cmo(self, data, period=14, price_col='close'):
        """
        Calculate Chande Momentum Oscillator for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            period (int): Period for CMO calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: CMO values
        """
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series()
            
        # Calculate price changes
        delta = data[price_col].diff()
        
        # Create gain and loss series
        gain = delta.copy()
        loss = delta.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate sum of gains and losses over the period
        gain_sum = gain.rolling(window=period).sum()
        loss_sum = loss.rolling(window=period).sum()
        
        # Calculate CMO
        cmo = 100 * ((gain_sum - loss_sum) / (gain_sum + loss_sum))
        
        return cmo
    
    def calculate_adaptive_moving_average(self, data, period=10, fast_period=2, slow_period=30, price_col='close'):
        """
        Calculate Kaufman's Adaptive Moving Average for the given data
        
        Args:
            data (pd.DataFrame): Price data with price column
            period (int): Period for efficiency ratio calculation
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: KAMA values
        """
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series()
            
        # Calculate price change and volatility
        change = abs(data[price_col] - data[price_col].shift(period))
        volatility = abs(data[price_col].diff()).rolling(window=period).sum()
        
        # Calculate efficiency ratio
        er = change / volatility
        er = er.replace([np.inf, -np.inf], 0)
        er = er.fillna(0)
        
        # Calculate smoothing constant
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        
        # Calculate adaptive smoothing constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        kama = pd.Series(data[price_col].iloc[0], index=data.index)
        
        # Calculate KAMA
        for i in range(1, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data[price_col].iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def calculate_fair_value_gap(self, data):
        """
        Calculate Fair Value Gap (FVG) for the given data.
        FVG occurs when price moves rapidly in one direction, leaving a gap between candles.
        
        Args:
            data (pd.DataFrame): Price data with OHLC columns
            
        Returns:
            pd.Series: Fair Value Gap values (positive for bullish gaps, negative for bearish gaps)
        """
        if data is None or data.empty or len(data) < 3:
            return pd.Series(dtype=float)
            
        # Initialize result series
        fvg = pd.Series(0.0, index=data.index)
        
        # Calculate FVG for each candle (starting from the 3rd candle)
        for i in range(2, len(data)):
            # Get current and previous 2 candles
            candle_1 = data.iloc[i-2]  # Oldest
            candle_2 = data.iloc[i-1]  # Middle
            candle_3 = data.iloc[i]    # Current
            
            # Check for bullish FVG (low of candle_3 > high of candle_1)
            if candle_3['low'] > candle_1['high']:
                # Calculate gap size (normalized by price)
                gap_size = (candle_3['low'] - candle_1['high']) / candle_2['close']
                fvg.iloc[i] = gap_size
                
            # Check for bearish FVG (high of candle_3 < low of candle_1)
            elif candle_3['high'] < candle_1['low']:
                # Calculate gap size (normalized by price, negative for bearish)
                gap_size = (candle_3['high'] - candle_1['low']) / candle_2['close']
                fvg.iloc[i] = gap_size
        
        return fvg
    
    def calculate_liquidity_zones(self, data):
        """
        Calculate liquidity zones (support and resistance) based on price action and volume.
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            dict: Dictionary with support and resistance zones and their strength
        """
        if data is None or data.empty or len(data) < 20:
            return {'support': pd.Series(), 'resistance': pd.Series()}
            
        # Initialize result series for support and resistance
        support = pd.Series(0.0, index=data.index)
        resistance = pd.Series(0.0, index=data.index)
        
        # Look for swing lows (potential support)
        for i in range(2, len(data) - 2):
            # Check if current low is lower than previous and next lows
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i-2] and
                data['low'].iloc[i] < data['low'].iloc[i+1] and 
                data['low'].iloc[i] < data['low'].iloc[i+2]):
                
                # Calculate strength based on volume and surrounding price action
                strength = 1.0
                
                # Higher volume increases strength
                if 'volume' in data.columns:
                    vol_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-10:i+10].mean()
                    strength *= min(vol_ratio, 3.0)  # Cap at 3x
                
                # Stronger if price bounced from this level before
                price_level = data['low'].iloc[i]
                nearby_levels = data['low'].iloc[max(0, i-20):i].tolist()
                for level in nearby_levels:
                    if abs(level - price_level) / price_level < 0.01:  # Within 1%
                        strength *= 1.2
                
                support.iloc[i] = strength
        
        # Look for swing highs (potential resistance)
        for i in range(2, len(data) - 2):
            # Check if current high is higher than previous and next highs
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i-2] and
                data['high'].iloc[i] > data['high'].iloc[i+1] and 
                data['high'].iloc[i] > data['high'].iloc[i+2]):
                
                # Calculate strength based on volume and surrounding price action
                strength = 1.0
                
                # Higher volume increases strength
                if 'volume' in data.columns:
                    vol_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-10:i+10].mean()
                    strength *= min(vol_ratio, 3.0)  # Cap at 3x
                
                # Stronger if price rejected from this level before
                price_level = data['high'].iloc[i]
                nearby_levels = data['high'].iloc[max(0, i-20):i].tolist()
                for level in nearby_levels:
                    if abs(level - price_level) / price_level < 0.01:  # Within 1%
                        strength *= 1.2
                
                resistance.iloc[i] = strength
        
        return {
            'support': support.astype(float),
            'resistance': resistance.astype(float)
        }
    
    def calculate_moving_averages(self, data, close_col='close', periods=[5, 10, 20, 50, 200]):
        """
        Calculate multiple moving averages and crossover signals.
        
        Args:
            data (pd.DataFrame): Price data with close column
            close_col (str): Column name for close price data
            periods (list): List of periods for moving average calculations
            
        Returns:
            dict: Dictionary with moving averages and crossover signals
        """
        if data is None or data.empty:
            return {'sma': {}, 'ema': {}, 'crossovers': {}}
            
        # Initialize result dictionaries
        sma_dict = {}
        ema_dict = {}
        crossovers = {}
        
        # Calculate simple moving averages for each period
        for period in periods:
            sma_dict[f'{period}'] = self.calculate_sma(data, period=period, price_col=close_col).astype(float)
            
        # Calculate exponential moving averages for each period
        for period in periods:
            ema_dict[f'{period}'] = self.calculate_ema(data, period=period, price_col=close_col).astype(float)
            
        # Calculate moving average crossovers
        if len(periods) >= 2:
            # Sort periods to ensure consistent crossover calculations
            sorted_periods = sorted(periods)
            
            for i in range(len(sorted_periods) - 1):
                short_period = sorted_periods[i]
                long_period = sorted_periods[i + 1]
                
                # Calculate crossover signals (1 for bullish crossover, -1 for bearish, 0 for no crossover)
                short_ma = ema_dict[f'{short_period}']
                long_ma = ema_dict[f'{long_period}']
                
                crossover = pd.Series(0, index=data.index)
                crossover[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1  # Bullish crossover
                crossover[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1  # Bearish crossover
                
                crossovers[f'{short_period}_{long_period}'] = crossover.astype(float)
        
        return {
            'sma': sma_dict,
            'ema': ema_dict,
            'crossovers': crossovers
        }
    
    def calculate_volatility(self, data, close_col='close', high_col='high', low_col='low', periods=[14, 30, 60]):
        """
        Calculate various volatility indicators.
        
        Args:
            data (pd.DataFrame): Price data with close, high, and low columns
            close_col (str): Column name for close price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            periods (list): List of periods for volatility calculations
            
        Returns:
            dict: Dictionary with volatility indicators
        """
        if data is None or data.empty:
            return {'historical': {}, 'atr': {}, 'regime': {}}
            
        # Initialize result dictionaries
        hist_vol = {}
        atr_dict = {}
        
        # Calculate daily returns
        returns = data[close_col].pct_change()
        
        # Calculate historical volatility (standard deviation of returns) for each period
        for period in periods:
            hist_vol[f'{period}'] = (returns.rolling(window=period).std() * np.sqrt(252)).astype(float)  # Annualized
        
        # Calculate Average True Range (ATR) for each period
        for period in periods:
            # Calculate True Range
            tr1 = data[high_col] - data[low_col]  # Current high - current low
            tr2 = abs(data[high_col] - data[close_col].shift(1))  # Current high - previous close
            tr3 = abs(data[low_col] - data[close_col].shift(1))  # Current low - previous close
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr_dict[f'{period}'] = true_range.rolling(window=period).mean().astype(float)
        
        # Calculate Bollinger Bands volatility (bandwidth)
        bb_period = 20
        if len(data) >= bb_period:
            sma = data[close_col].rolling(window=bb_period).mean()
            std_dev = data[close_col].rolling(window=bb_period).std()
            
            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)
            
            # Calculate bandwidth: (upper - lower) / middle
            bb_bandwidth = ((upper_band - lower_band) / sma).astype(float)
        else:
            bb_bandwidth = pd.Series(dtype=float)
        
        # Calculate Chaikin Volatility
        chaikin_period = 10
        if len(data) >= chaikin_period:
            high_low_diff = data[high_col] - data[low_col]
            ema_high_low = high_low_diff.ewm(span=chaikin_period, adjust=False).mean()
            chaikin_volatility = ((ema_high_low / ema_high_low.shift(chaikin_period) - 1) * 100).astype(float)
        else:
            chaikin_volatility = pd.Series(dtype=float)
        
        # Determine volatility regime (high/normal/low)
        # Use the 30-period historical volatility as reference
        if f'30' in hist_vol:
            vol_30 = hist_vol['30']
            
            # Calculate long-term average and standard deviation of volatility
            vol_avg = vol_30.rolling(window=252).mean()
            vol_std = vol_30.rolling(window=252).std()
            
            # Initialize regime series
            regime = pd.Series(0, index=data.index)  # 0 = normal, 1 = high, -1 = low
            
            # High volatility: > 1 standard deviation above average
            regime[vol_30 > (vol_avg + vol_std)] = 1
            
            # Low volatility: > 1 standard deviation below average
            regime[vol_30 < (vol_avg - vol_std)] = -1
            
            regime = regime.astype(float)
        else:
            regime = pd.Series(dtype=float)
        
        return {
            'historical': hist_vol,
            'atr': atr_dict,
            'bb_bandwidth': bb_bandwidth,
            'chaikin_volatility': chaikin_volatility,
            'regime': regime
        }
