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
    
    # New indicator methods for Phase 1 enhancements
    
    def chande_momentum_oscillator(self, period=14, price_col='close'):
        """
        Calculate Chande Momentum Oscillator using instance data
        
        Args:
            period (int): Period for CMO calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: CMO values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_cmo(self.data, period, price_col)
    
    def stochastic_rsi(self, period=14, smooth_k=3, smooth_d=3, price_col='close'):
        """
        Calculate Stochastic RSI using instance data
        
        Args:
            period (int): Period for RSI calculation
            smooth_k (int): Smoothing for %K line
            smooth_d (int): Smoothing for %D line
            price_col (str): Column name for price data
            
        Returns:
            pd.DataFrame: DataFrame with %K and %D values
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        return self.calculate_stoch_rsi(self.data, period, smooth_k, smooth_d, price_col)
    
    def on_balance_volume(self, price_col='close'):
        """
        Calculate On-Balance Volume using instance data
        
        Args:
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: OBV values
        """
        if self.data is None or self.data.empty or 'volume' not in self.data.columns:
            return pd.Series()
            
        return self.calculate_obv(self.data, price_col)
    
    def accumulation_distribution_line(self):
        """
        Calculate Accumulation/Distribution Line using instance data
        
        Returns:
            pd.Series: A/D Line values
        """
        if self.data is None or self.data.empty or not all(col in self.data.columns for col in ['high', 'low', 'close', 'volume']):
            return pd.Series()
            
        return self.calculate_adl(self.data)
    
    def adaptive_moving_average(self, er_period=10, fast_period=2, slow_period=30, price_col='close'):
        """
        Calculate Adaptive Moving Average using instance data
        
        Args:
            er_period (int): Period for Efficiency Ratio calculation
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: AMA values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_ama(self.data, er_period, fast_period, slow_period, price_col)
    
    def volatility_regime(self, lookback=20, percentile_threshold=75, price_col='close'):
        """
        Identify volatility regime using instance data
        
        Args:
            lookback (int): Period for volatility calculation
            percentile_threshold (int): Percentile threshold for regime classification
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: Volatility regime classification ('high', 'normal', 'low')
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.identify_volatility_regime(self.data, lookback, percentile_threshold, price_col)
    
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
    
    @staticmethod
    def calculate_imi(data, period=14):
        """
        Calculate Intraday Momentum Index (IMI)
        
        Args:
            data (pd.DataFrame): Historical price data with OHLC
            period (int): Period for IMI calculation
            
        Returns:
            pd.Series: IMI values
        """
        if data.empty or not all(col in data.columns for col in ['open', 'close']):
            return pd.Series()
        
        # Calculate gains and losses
        gains = np.where(data['close'] > data['open'], data['close'] - data['open'], 0)
        losses = np.where(data['open'] > data['close'], data['open'] - data['close'], 0)
        
        # Convert to Series
        gains_series = pd.Series(gains, index=data.index if hasattr(data, 'index') else None)
        losses_series = pd.Series(losses, index=data.index if hasattr(data, 'index') else None)
        
        # Calculate sum of gains and losses over period
        sum_gains = gains_series.rolling(window=period).sum()
        sum_losses = losses_series.rolling(window=period).sum()
        
        # Calculate IMI
        imi = 100 * (sum_gains / (sum_gains + sum_losses))
        
        return imi
    
    @staticmethod
    def calculate_mfi(data, period=14):
        """
        Calculate Money Flow Index (MFI)
        
        Args:
            data (pd.DataFrame): Historical price data with OHLC and volume
            period (int): Period for MFI calculation
            
        Returns:
            pd.Series: MFI values
        """
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            return pd.Series()
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * data['volume']
        
        # Get price change direction
        price_change = typical_price.diff()
        
        # Separate positive and negative money flow
        positive_flow = np.where(price_change > 0, raw_money_flow, 0)
        negative_flow = np.where(price_change < 0, raw_money_flow, 0)
        
        # Convert to Series
        positive_flow_series = pd.Series(positive_flow, index=data.index if hasattr(data, 'index') else None)
        negative_flow_series = pd.Series(negative_flow, index=data.index if hasattr(data, 'index') else None)
        
        # Calculate sum over period
        positive_sum = positive_flow_series.rolling(window=period).sum()
        negative_sum = negative_flow_series.rolling(window=period).sum()
        
        # Calculate money flow ratio
        money_flow_ratio = positive_sum / negative_sum
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    @staticmethod
    def calculate_fair_value_gap(data):
        """
        Calculate Fair Value Gap (FVG)
        
        Args:
            data (pd.DataFrame): Historical price data with OHLC
            
        Returns:
            pd.DataFrame: DataFrame with FVG information
        """
        if data.empty or not all(col in data.columns for col in ['high', 'low']):
            return pd.DataFrame()
        
        # Initialize results
        fvg_data = []
        
        # Need at least 3 candles to identify FVG
        for i in range(2, len(data)):
            # Bullish FVG: Candle 1 high < Candle 3 low
            if data['high'].iloc[i-2] < data['low'].iloc[i]:
                fvg_data.append({
                    'datetime': data.index[i] if hasattr(data, 'index') else i,
                    'type': 'bullish',
                    'top': data['low'].iloc[i],
                    'bottom': data['high'].iloc[i-2],
                    'size': data['low'].iloc[i] - data['high'].iloc[i-2]
                })
            
            # Bearish FVG: Candle 1 low > Candle 3 high
            if data['low'].iloc[i-2] > data['high'].iloc[i]:
                fvg_data.append({
                    'datetime': data.index[i] if hasattr(data, 'index') else i,
                    'type': 'bearish',
                    'top': data['low'].iloc[i-2],
                    'bottom': data['high'].iloc[i],
                    'size': data['low'].iloc[i-2] - data['high'].iloc[i]
                })
        
        return pd.DataFrame(fvg_data)
    
    @staticmethod
    def calculate_liquidity_zones(data, volume_threshold=0.8):
        """
        Calculate Liquidity Zones (high volume support/resistance)
        
        Args:
            data (pd.DataFrame): Historical price data with OHLC and volume
            volume_threshold (float): Percentile threshold for high volume (0-1)
            
        Returns:
            pd.DataFrame: DataFrame with liquidity zones
        """
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'volume']):
            return pd.DataFrame()
        
        # Calculate high volume threshold
        high_volume = data['volume'].quantile(volume_threshold)
        
        # Filter for high volume candles
        high_vol_data = data[data['volume'] >= high_volume]
        
        if high_vol_data.empty:
            return pd.DataFrame()
        
        # Initialize results
        liquidity_zones = []
        
        # Process high volume candles to identify liquidity zones
        for idx, row in high_vol_data.iterrows():
            # Check if this price level is already in a zone
            new_zone = True
            zone_idx = -1
            
            for i, zone in enumerate(liquidity_zones):
                # If price range overlaps with existing zone
                if (row['low'] <= zone['high'] and row['high'] >= zone['low']):
                    new_zone = False
                    zone_idx = i
                    break
            
            if new_zone:
                # Create new zone
                liquidity_zones.append({
                    'low': row['low'],
                    'high': row['high'],
                    'volume': row['volume'],
                    'count': 1,
                    'last_touch': idx
                })
            else:
                # Update existing zone
                zone = liquidity_zones[zone_idx]
                zone['low'] = min(zone['low'], row['low'])
                zone['high'] = max(zone['high'], row['high'])
                zone['volume'] += row['volume']
                zone['count'] += 1
                zone['last_touch'] = idx
        
        # Convert to DataFrame
        zones_df = pd.DataFrame(liquidity_zones)
        
        # Calculate zone strength based on volume and touch count
        if not zones_df.empty:
            zones_df['strength'] = zones_df['volume'] * zones_df['count']
            zones_df = zones_df.sort_values('strength', ascending=False)
        
        return zones_df
    
    @staticmethod
    def calculate_moving_averages(data, periods=[20, 50, 200], price_col='close'):
        """
        Calculate Simple Moving Averages (SMA)
        
        Args:
            data (pd.DataFrame): Historical price data
            periods (list): List of periods for SMA calculation
            price_col (str): Column name for price data
            
        Returns:
            dict: Dictionary of SMAs for each period
        """
        if data.empty:
            return {}
        
        sma_dict = {}
        
        for period in periods:
            if len(data) >= period:
                sma = data[price_col].rolling(window=period).mean()
                sma_dict[f'SMA_{period}'] = sma
        
        return sma_dict
    
    @staticmethod
    def calculate_volatility(data, period=20, price_col='close'):
        """
        Calculate historical volatility
        
        Args:
            data (pd.DataFrame): Historical price data
            period (int): Period for volatility calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: Annualized volatility
        """
        if data.empty or len(data) < period:
            return pd.Series()
        
        # Calculate log returns
        log_returns = np.log(data[price_col] / data[price_col].shift(1))
        
        # Calculate rolling standard deviation of log returns
        rolling_std = log_returns.rolling(window=period).std()
        
        # Annualize (assuming daily data, multiply by sqrt(252))
        annualized_vol = rolling_std * np.sqrt(252)
        
        return annualized_vol
    
    # New static methods for Phase 1 enhancements
    
    @staticmethod
    def calculate_cmo(data, period=14, price_col='close'):
        """
        Calculate Chande Momentum Oscillator (CMO)
        
        Args:
            data (pd.DataFrame): Historical price data
            period (int): Period for CMO calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: CMO values
        """
        if data.empty or len(data) < period + 1:
            return pd.Series()
        
        # Calculate price changes
        delta = data[price_col].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate sum of gains and losses over period
        sum_gain = gain.rolling(window=period).sum()
        sum_loss = loss.rolling(window=period).sum()
        
        # Initialize CMO series
        cmo = pd.Series(index=data.index)
        
        # Calculate CMO with handling for zero denominators
        valid_idx = (sum_gain + sum_loss) > 0
        cmo[valid_idx] = 100 * ((sum_gain - sum_loss) / (sum_gain + sum_loss))[valid_idx]
        
        # Handle zero denominator case (no price changes in period)
        cmo[(sum_gain + sum_loss) == 0] = 0
        
        return cmo
    
    @staticmethod
    def calculate_stoch_rsi(data, period=14, smooth_k=3, smooth_d=3, price_col='close'):
        """
        Calculate Stochastic RSI
        
        Args:
            data (pd.DataFrame): Historical price data
            period (int): Period for RSI and Stochastic calculation
            smooth_k (int): Smoothing for %K line
            smooth_d (int): Smoothing for %D line
            price_col (str): Column name for price data
            
        Returns:
            pd.DataFrame: DataFrame with %K and %D values
        """
        if data.empty or len(data) < period * 2:  # Need more data for RSI and then Stochastic
            return pd.DataFrame()
        
        # Calculate RSI
        delta = data[price_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Handle division by zero in RSI calculation
        rs = pd.Series(index=data.index)
        valid_idx = avg_loss > 0
        rs[valid_idx] = avg_gain[valid_idx] / avg_loss[valid_idx]
        rs[avg_loss == 0] = 100  # If no losses, RSI should be 100
        
        rsi = 100 - (100 / (1 + rs))
        
        # Apply Stochastic formula to RSI
        stoch_rsi = pd.Series(index=rsi.index)
        
        for i in range(period, len(rsi)):
            rsi_window = rsi.iloc[i-period+1:i+1]
            if not rsi_window.isna().all():  # Check if all values are NaN
                rsi_min = rsi_window.min()
                rsi_max = rsi_window.max()
                
                if not np.isnan(rsi_min) and not np.isnan(rsi_max) and (rsi_max - rsi_min) != 0:
                    stoch_rsi.iloc[i] = (rsi.iloc[i] - rsi_min) / (rsi_max - rsi_min)
                else:
                    stoch_rsi.iloc[i] = 0.5  # Default to middle if no range
        
        # Calculate %K (smoothed stoch_rsi)
        k = stoch_rsi.rolling(window=smooth_k).mean() * 100
        
        # Calculate %D (smoothed %K)
        d = k.rolling(window=smooth_d).mean()
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'k': k,
            'd': d
        }, index=data.index if hasattr(data, 'index') else None)
        
        return result
    
    @staticmethod
    def calculate_obv(data, price_col='close'):
        """
        Calculate On-Balance Volume (OBV)
        
        Args:
            data (pd.DataFrame): Historical price data with volume
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: OBV values
        """
        if data.empty or 'volume' not in data.columns:
            return pd.Series()
        
        # Initialize OBV series
        obv = pd.Series(0, index=data.index)
        
        # Calculate OBV
        for i in range(1, len(data)):
            if data[price_col].iloc[i] > data[price_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data[price_col].iloc[i] < data[price_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_adl(data):
        """
        Calculate Accumulation/Distribution Line
        
        Args:
            data (pd.DataFrame): Historical price data with OHLC and volume
            
        Returns:
            pd.Series: A/D Line values
        """
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            return pd.Series()
        
        # Calculate Money Flow Multiplier
        high_low_range = data['high'] - data['low']
        
        # Initialize A/D Line series
        adl = pd.Series(0, index=data.index)
        
        # Calculate A/D Line
        for i in range(1, len(data)):
            # Skip calculation if high equals low (avoid division by zero)
            if high_low_range.iloc[i] == 0:
                adl.iloc[i] = adl.iloc[i-1]
                continue
                
            # Calculate Money Flow Multiplier
            mfm = ((data['close'].iloc[i] - data['low'].iloc[i]) - 
                   (data['high'].iloc[i] - data['close'].iloc[i])) / high_low_range.iloc[i]
            
            # Calculate Money Flow Volume
            mfv = mfm * data['volume'].iloc[i]
            
            # Update A/D Line
            adl.iloc[i] = adl.iloc[i-1] + mfv
        
        return adl
    
    @staticmethod
    def calculate_ama(data, er_period=10, fast_period=2, slow_period=30, price_col='close'):
        """
        Calculate Adaptive Moving Average (AMA)
        
        Args:
            data (pd.DataFrame): Historical price data
            er_period (int): Period for Efficiency Ratio calculation
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: AMA values
        """
        if data.empty or len(data) < er_period + 1:
            return pd.Series()
        
        # Initialize AMA series
        ama = pd.Series(index=data.index)
        ama.iloc[0] = data[price_col].iloc[0]
        
        # Calculate AMA
        for i in range(1, len(data)):
            if i < er_period:
                ama.iloc[i] = data[price_col].iloc[i]  # Use price until we have enough data for ER
            else:
                # Calculate direction (absolute price change over er_period)
                direction = abs(data[price_col].iloc[i] - data[price_col].iloc[i-er_period])
                
                # Calculate volatility (sum of absolute price changes over er_period)
                volatility = 0
                for j in range(i-er_period+1, i+1):
                    volatility += abs(data[price_col].iloc[j] - data[price_col].iloc[j-1])
                
                # Calculate Efficiency Ratio (ER)
                er = 0 if volatility == 0 else direction / volatility
                
                # Calculate smoothing constant
                fast_sc = 2.0 / (fast_period + 1)
                slow_sc = 2.0 / (slow_period + 1)
                sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
                
                # Calculate AMA
                ama.iloc[i] = ama.iloc[i-1] + sc * (data[price_col].iloc[i] - ama.iloc[i-1])
        
        return ama
    
    @staticmethod
    def identify_volatility_regime(data, lookback=20, percentile_threshold=75, price_col='close'):
        """
        Identify volatility regime (high, normal, low)
        
        Args:
            data (pd.DataFrame): Historical price data
            lookback (int): Period for volatility calculation
            percentile_threshold (int): Percentile threshold for regime classification
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: Volatility regime classification ('high', 'normal', 'low')
        """
        if data.empty or len(data) < lookback + 20:  # Need enough data for meaningful percentiles
            return pd.Series()
        
        # Calculate returns
        returns = data[price_col].pct_change().fillna(0)
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=lookback).std() * np.sqrt(252)  # Annualized
        
        # Initialize regime series with 'normal' as default
        regime = pd.Series('normal', index=data.index)
        
        # Need at least 100 data points to calculate meaningful percentiles
        if len(data) >= 100:
            # Calculate rolling percentile using expanding window
            for i in range(lookback + 20, len(data)):
                vol_window = rolling_vol.iloc[:i]
                current_vol = rolling_vol.iloc[i]
                
                if not np.isnan(current_vol):
                    percentile = stats.percentileofscore(vol_window.dropna(), current_vol)
                    
                    if percentile >= percentile_threshold:
                        regime.iloc[i] = 'high'
                    elif percentile <= (100 - percentile_threshold):
                        regime.iloc[i] = 'low'
                    else:
                        regime.iloc[i] = 'normal'
        
        return regime
