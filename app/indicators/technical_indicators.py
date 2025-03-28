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
        Calculate Kaufman's Adaptive Moving Average using instance data
        
        Args:
            er_period (int): Period for efficiency ratio calculation
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: KAMA values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_ama(self.data, er_period, fast_period, slow_period, price_col)
    
    # New methods to implement missing indicators
    
    def atr(self, high_col='high', low_col='low', close_col='close', period=14):
        """
        Calculate Average True Range using instance data
        
        Args:
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            period (int): Period for ATR calculation
            
        Returns:
            pd.Series: ATR values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_atr(self.data, high_col, low_col, close_col, period)
    
    def adx(self, high_col='high', low_col='low', close_col='close', period=14):
        """
        Calculate Average Directional Index using instance data
        
        Args:
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            period (int): Period for ADX calculation
            
        Returns:
            pd.DataFrame: DataFrame with ADX, +DI, and -DI values
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        return self.calculate_adx(self.data, high_col, low_col, close_col, period)
    
    def stochastic(self, high_col='high', low_col='low', close_col='close', k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator using instance data
        
        Args:
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            k_period (int): Period for %K calculation
            d_period (int): Period for %D calculation
            
        Returns:
            pd.DataFrame: DataFrame with %K and %D values
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        return self.calculate_stochastic(self.data, high_col, low_col, close_col, k_period, d_period)
    
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
            
        return self.calculate_obv(self.data, close_col, volume_col)
    
    def cmf(self, high_col='high', low_col='low', close_col='close', volume_col='volume', period=20):
        """
        Calculate Chaikin Money Flow using instance data
        
        Args:
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            period (int): Period for CMF calculation
            
        Returns:
            pd.Series: CMF values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_cmf(self.data, high_col, low_col, close_col, volume_col, period)
    
    def mfi(self, high_col='high', low_col='low', close_col='close', volume_col='volume', period=14):
        """
        Calculate Money Flow Index using instance data
        
        Args:
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            period (int): Period for MFI calculation
            
        Returns:
            pd.Series: MFI values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_mfi(self.data, high_col, low_col, close_col, volume_col, period)
    
    def cci(self, high_col='high', low_col='low', close_col='close', period=20):
        """
        Calculate Commodity Channel Index using instance data
        
        Args:
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            period (int): Period for CCI calculation
            
        Returns:
            pd.Series: CCI values
        """
        if self.data is None or self.data.empty:
            return pd.Series()
            
        return self.calculate_cci(self.data, high_col, low_col, close_col, period)
    
    @staticmethod
    def calculate_rsi(data, period=14, price_col='close'):
        """
        Calculate Relative Strength Index
        
        Args:
            data (pd.DataFrame): Price data
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
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
        """
        Calculate Moving Average Convergence Divergence
        
        Args:
            data (pd.DataFrame): Price data
            fast_period (int): Period for fast EMA
            slow_period (int): Period for slow EMA
            signal_period (int): Period for signal line
            price_col (str): Column name for price data
            
        Returns:
            tuple: (MACD line, signal line, histogram)
        """
        if data is None or data.empty or len(data) < slow_period:
            return pd.Series(), pd.Series(), pd.Series()
            
        # Calculate fast and slow EMAs
        fast_ema = data[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_sma(data, period=20, price_col='close'):
        """
        Calculate Simple Moving Average
        
        Args:
            data (pd.DataFrame): Price data
            period (int): Period for SMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: SMA values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        return data[price_col].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data, period=20, price_col='close'):
        """
        Calculate Exponential Moving Average
        
        Args:
            data (pd.DataFrame): Price data
            period (int): Period for EMA calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: EMA values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        return data[price_col].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2, price_col='close'):
        """
        Calculate Bollinger Bands
        
        Args:
            data (pd.DataFrame): Price data
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
        })
        
        return result
    
    @staticmethod
    def calculate_imi(data, period=14):
        """
        Calculate Intraday Momentum Index
        
        Args:
            data (pd.DataFrame): Price data
            period (int): Period for IMI calculation
            
        Returns:
            pd.Series: IMI values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate gains and losses
        gains = np.where(data['close'] > data['open'], data['close'] - data['open'], 0)
        losses = np.where(data['open'] > data['close'], data['open'] - data['close'], 0)
        
        # Calculate sum of gains and losses over period
        sum_gains = pd.Series(gains).rolling(window=period).sum()
        sum_losses = pd.Series(losses).rolling(window=period).sum()
        
        # Calculate IMI
        imi = 100 * (sum_gains / (sum_gains + sum_losses))
        
        return imi
    
    @staticmethod
    def calculate_mfi(data, high_col='high', low_col='low', close_col='close', volume_col='volume', period=14):
        """
        Calculate Money Flow Index
        
        Args:
            data (pd.DataFrame): Price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            period (int): Period for MFI calculation
            
        Returns:
            pd.Series: MFI values
        """
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series()
            
        # Calculate typical price
        typical_price = (data[high_col] + data[low_col] + data[close_col]) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * data[volume_col]
        
        # Calculate money flow direction
        money_flow_positive = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        money_flow_negative = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
        
        # Calculate money flow ratio
        positive_flow = pd.Series(money_flow_positive).rolling(window=period).sum()
        negative_flow = pd.Series(money_flow_negative).rolling(window=period).sum()
        
        money_flow_ratio = positive_flow / negative_flow
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    @staticmethod
    def calculate_fair_value_gap(data):
        """
        Identify Fair Value Gaps in price data
        
        Args:
            data (pd.DataFrame): Price data with OHLC
            
        Returns:
            pd.DataFrame: DataFrame with identified fair value gaps
        """
        if data is None or data.empty or len(data) < 3:
            return pd.DataFrame()
            
        # Initialize results
        bullish_fvg = []
        bearish_fvg = []
        
        # Loop through candles to identify FVGs
        for i in range(2, len(data)):
            # Bullish FVG: Current candle's low > Previous candle's high
            if data['low'].iloc[i] > data['high'].iloc[i-1]:
                bullish_fvg.append({
                    'date': data.index[i] if hasattr(data, 'index') else i,
                    'type': 'bullish',
                    'top': data['low'].iloc[i],
                    'bottom': data['high'].iloc[i-1],
                    'size': data['low'].iloc[i] - data['high'].iloc[i-1]
                })
                
            # Bearish FVG: Current candle's high < Previous candle's low
            if data['high'].iloc[i] < data['low'].iloc[i-1]:
                bearish_fvg.append({
                    'date': data.index[i] if hasattr(data, 'index') else i,
                    'type': 'bearish',
                    'top': data['low'].iloc[i-1],
                    'bottom': data['high'].iloc[i],
                    'size': data['low'].iloc[i-1] - data['high'].iloc[i]
                })
                
        # Combine results
        all_fvg = bullish_fvg + bearish_fvg
        
        if not all_fvg:
            return pd.DataFrame()
            
        return pd.DataFrame(all_fvg)
    
    @staticmethod
    def calculate_liquidity_zones(data, volume_threshold=0.8):
        """
        Identify liquidity zones based on volume profile
        
        Args:
            data (pd.DataFrame): Price data with OHLC and volume
            volume_threshold (float): Threshold for high volume (percentile)
            
        Returns:
            pd.DataFrame: DataFrame with identified liquidity zones
        """
        if data is None or data.empty or 'volume' not in data.columns:
            return pd.DataFrame()
            
        # Calculate price levels
        price_levels = []
        for i in range(len(data)):
            # Add high and low to price levels
            price_levels.append({
                'price': data['high'].iloc[i],
                'volume': data['volume'].iloc[i],
                'date': data.index[i] if hasattr(data, 'index') else i
            })
            
            price_levels.append({
                'price': data['low'].iloc[i],
                'volume': data['volume'].iloc[i],
                'date': data.index[i] if hasattr(data, 'index') else i
            })
            
        # Convert to DataFrame
        price_df = pd.DataFrame(price_levels)
        
        # Group by price and sum volume
        price_volume = price_df.groupby('price')['volume'].sum().reset_index()
        
        # Sort by volume
        price_volume = price_volume.sort_values('volume', ascending=False)
        
        # Calculate cumulative volume percentage
        total_volume = price_volume['volume'].sum()
        price_volume['volume_pct'] = price_volume['volume'] / total_volume
        price_volume['cum_volume_pct'] = price_volume['volume_pct'].cumsum()
        
        # Identify high volume nodes (liquidity zones)
        high_volume_nodes = price_volume[price_volume['cum_volume_pct'] <= volume_threshold]
        
        # Sort by price
        high_volume_nodes = high_volume_nodes.sort_values('price')
        
        # Group adjacent price levels
        if high_volume_nodes.empty:
            return pd.DataFrame()
            
        zones = []
        current_zone = {
            'start_price': high_volume_nodes['price'].iloc[0],
            'end_price': high_volume_nodes['price'].iloc[0],
            'total_volume': high_volume_nodes['volume'].iloc[0]
        }
        
        for i in range(1, len(high_volume_nodes)):
            current_price = high_volume_nodes['price'].iloc[i]
            prev_price = high_volume_nodes['price'].iloc[i-1]
            
            # If prices are close, extend current zone
            if abs(current_price - prev_price) / prev_price < 0.01:  # 1% threshold
                current_zone['end_price'] = current_price
                current_zone['total_volume'] += high_volume_nodes['volume'].iloc[i]
            else:
                # Add current zone to results and start a new one
                zones.append(current_zone)
                current_zone = {
                    'start_price': current_price,
                    'end_price': current_price,
                    'total_volume': high_volume_nodes['volume'].iloc[i]
                }
                
        # Add the last zone
        zones.append(current_zone)
        
        # Calculate zone strength based on volume
        zones_df = pd.DataFrame(zones)
        zones_df['strength'] = zones_df['total_volume'] / zones_df['total_volume'].sum()
        
        return zones_df
    
    @staticmethod
    def calculate_moving_averages(data, periods=[20, 50, 200], price_col='close'):
        """
        Calculate multiple moving averages
        
        Args:
            data (pd.DataFrame): Price data
            periods (list): List of periods for moving averages
            price_col (str): Column name for price data
            
        Returns:
            pd.DataFrame: DataFrame with moving averages for each period
        """
        if data is None or data.empty:
            return pd.DataFrame()
            
        result = pd.DataFrame(index=data.index if hasattr(data, 'index') else None)
        
        for period in periods:
            if len(data) >= period:
                result[f'ma_{period}'] = data[price_col].rolling(window=period).mean()
                
        return result
    
    @staticmethod
    def calculate_volatility(data, period=20, price_col='close'):
        """
        Calculate historical volatility
        
        Args:
            data (pd.DataFrame): Price data
            period (int): Period for volatility calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: Volatility values (annualized)
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate log returns
        log_returns = np.log(data[price_col] / data[price_col].shift(1))
        
        # Calculate rolling standard deviation
        rolling_std = log_returns.rolling(window=period).std()
        
        # Annualize (assuming daily data, multiply by sqrt(252))
        annualized_volatility = rolling_std * np.sqrt(252)
        
        return annualized_volatility
    
    @staticmethod
    def calculate_cmo(data, period=14, price_col='close'):
        """
        Calculate Chande Momentum Oscillator
        
        Args:
            data (pd.DataFrame): Price data
            period (int): Period for CMO calculation
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: CMO values
        """
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series()
            
        # Calculate price changes
        delta = data[price_col].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate sum of gains and losses over period
        sum_gain = gain.rolling(window=period).sum()
        sum_loss = loss.rolling(window=period).sum()
        
        # Calculate CMO
        cmo = 100 * ((sum_gain - sum_loss) / (sum_gain + sum_loss))
        
        return cmo
    
    @staticmethod
    def calculate_stoch_rsi(data, period=14, smooth_k=3, smooth_d=3, price_col='close'):
        """
        Calculate Stochastic RSI
        
        Args:
            data (pd.DataFrame): Price data
            period (int): Period for RSI and Stochastic calculation
            smooth_k (int): Smoothing for %K line
            smooth_d (int): Smoothing for %D line
            price_col (str): Column name for price data
            
        Returns:
            pd.DataFrame: DataFrame with %K and %D values
        """
        if data is None or data.empty or len(data) < period * 2:
            return pd.DataFrame()
            
        # Calculate RSI
        rsi = TechnicalIndicators.calculate_rsi(data, period, price_col)
        
        # Calculate Stochastic RSI
        stoch_rsi = pd.Series(index=rsi.index)
        
        for i in range(period, len(rsi)):
            rsi_window = rsi.iloc[i-period+1:i+1]
            
            if not rsi_window.empty:
                min_rsi = rsi_window.min()
                max_rsi = rsi_window.max()
                
                if max_rsi - min_rsi != 0:
                    stoch_rsi.iloc[i] = (rsi.iloc[i] - min_rsi) / (max_rsi - min_rsi)
                else:
                    stoch_rsi.iloc[i] = 0.5  # Default to middle if range is zero
            
        # Apply smoothing to %K
        k = stoch_rsi.rolling(window=smooth_k).mean() * 100
        
        # Calculate %D (SMA of %K)
        d = k.rolling(window=smooth_d).mean()
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'k': k,
            'd': d
        })
        
        return result
    
    @staticmethod
    def calculate_obv(data, close_col='close', volume_col='volume'):
        """
        Calculate On-Balance Volume
        
        Args:
            data (pd.DataFrame): Price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            
        Returns:
            pd.Series: OBV values
        """
        if data is None or data.empty or volume_col not in data.columns:
            return pd.Series()
            
        # Initialize OBV with first volume value
        obv = pd.Series(index=data.index if hasattr(data, 'index') else None)
        obv.iloc[0] = data[volume_col].iloc[0]
        
        # Calculate OBV
        for i in range(1, len(data)):
            if data[close_col].iloc[i] > data[close_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data[volume_col].iloc[i]
            elif data[close_col].iloc[i] < data[close_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data[volume_col].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    @staticmethod
    def calculate_adl(data):
        """
        Calculate Accumulation/Distribution Line
        
        Args:
            data (pd.DataFrame): Price data with OHLC and volume
            
        Returns:
            pd.Series: A/D Line values
        """
        if data is None or data.empty or not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            return pd.Series()
            
        # Calculate Money Flow Multiplier
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mfv = mfm * data['volume']
        
        # Calculate A/D Line (cumulative sum of Money Flow Volume)
        adl = mfv.cumsum()
        
        return adl
    
    @staticmethod
    def calculate_ama(data, er_period=10, fast_period=2, slow_period=30, price_col='close'):
        """
        Calculate Kaufman's Adaptive Moving Average
        
        Args:
            data (pd.DataFrame): Price data
            er_period (int): Period for efficiency ratio calculation
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            price_col (str): Column name for price data
            
        Returns:
            pd.Series: KAMA values
        """
        if data is None or data.empty or len(data) < er_period + 1:
            return pd.Series()
            
        # Calculate price change
        change = abs(data[price_col] - data[price_col].shift(er_period))
        
        # Calculate volatility (sum of absolute price changes over er_period)
        volatility = pd.Series(index=data.index if hasattr(data, 'index') else None)
        
        for i in range(er_period, len(data)):
            volatility.iloc[i] = sum(abs(data[price_col].iloc[i-er_period+1:i+1].diff().dropna()))
            
        # Calculate efficiency ratio
        er = change / volatility
        er = er.replace([np.inf, -np.inf], 0)  # Handle division by zero
        
        # Calculate smoothing constant
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        
        # Calculate adaptive smoothing constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        kama = pd.Series(index=data.index if hasattr(data, 'index') else None)
        kama.iloc[er_period] = data[price_col].iloc[er_period]  # Initialize with price
        
        for i in range(er_period + 1, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data[price_col].iloc[i] - kama.iloc[i-1])
            
        return kama
    
    # New static methods for missing indicators
    
    @staticmethod
    def calculate_atr(data, high_col='high', low_col='low', close_col='close', period=14):
        """
        Calculate Average True Range
        
        Args:
            data (pd.DataFrame): Price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            period (int): Period for ATR calculation
            
        Returns:
            pd.Series: ATR values
        """
        if data is None or data.empty or len(data) < period + 1:
            return pd.Series()
            
        # Calculate True Range
        tr1 = data[high_col] - data[low_col]  # Current high - current low
        tr2 = abs(data[high_col] - data[close_col].shift(1))  # Current high - previous close
        tr3 = abs(data[low_col] - data[close_col].shift(1))  # Current low - previous close
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR (simple moving average of True Range)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_adx(data, high_col='high', low_col='low', close_col='close', period=14):
        """
        Calculate Average Directional Index
        
        Args:
            data (pd.DataFrame): Price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            period (int): Period for ADX calculation
            
        Returns:
            pd.DataFrame: DataFrame with ADX, +DI, and -DI values
        """
        if data is None or data.empty or len(data) < period * 2:
            return pd.DataFrame()
            
        # Calculate True Range
        tr1 = data[high_col] - data[low_col]  # Current high - current low
        tr2 = abs(data[high_col] - data[close_col].shift(1))  # Current high - previous close
        tr3 = abs(data[low_col] - data[close_col].shift(1))  # Current low - previous close
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = data[high_col] - data[high_col].shift(1)
        down_move = data[low_col].shift(1) - data[low_col]
        
        # Calculate +DM and -DM
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate smoothed +DM, -DM, and TR
        plus_di = pd.Series(plus_dm).rolling(window=period).sum() / tr.rolling(window=period).sum() * 100
        minus_di = pd.Series(minus_dm).rolling(window=period).sum() / tr.rolling(window=period).sum() * 100
        
        # Calculate Directional Index (DX)
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        
        # Calculate ADX (smoothed DX)
        adx = dx.rolling(window=period).mean()
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'adx': adx,
            'di_plus': plus_di,
            'di_minus': minus_di
        })
        
        return result
    
    @staticmethod
    def calculate_stochastic(data, high_col='high', low_col='low', close_col='close', k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            data (pd.DataFrame): Price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            k_period (int): Period for %K calculation
            d_period (int): Period for %D calculation
            
        Returns:
            pd.DataFrame: DataFrame with %K and %D values
        """
        if data is None or data.empty or len(data) < k_period:
            return pd.DataFrame()
            
        # Calculate %K
        lowest_low = data[low_col].rolling(window=k_period).min()
        highest_high = data[high_col].rolling(window=k_period).max()
        
        k = 100 * ((data[close_col] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (SMA of %K)
        d = k.rolling(window=d_period).mean()
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'k': k,
            'd': d
        })
        
        return result
    
    @staticmethod
    def calculate_cmf(data, high_col='high', low_col='low', close_col='close', volume_col='volume', period=20):
        """
        Calculate Chaikin Money Flow
        
        Args:
            data (pd.DataFrame): Price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            volume_col (str): Column name for volume data
            period (int): Period for CMF calculation
            
        Returns:
            pd.Series: CMF values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate Money Flow Multiplier
        mfm = ((data[close_col] - data[low_col]) - (data[high_col] - data[close_col])) / (data[high_col] - data[low_col])
        mfm = mfm.replace([np.inf, -np.inf], 0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mfv = mfm * data[volume_col]
        
        # Calculate Chaikin Money Flow
        cmf = mfv.rolling(window=period).sum() / data[volume_col].rolling(window=period).sum()
        
        return cmf
    
    @staticmethod
    def calculate_cci(data, high_col='high', low_col='low', close_col='close', period=20):
        """
        Calculate Commodity Channel Index
        
        Args:
            data (pd.DataFrame): Price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            close_col (str): Column name for close price data
            period (int): Period for CCI calculation
            
        Returns:
            pd.Series: CCI values
        """
        if data is None or data.empty or len(data) < period:
            return pd.Series()
            
        # Calculate Typical Price
        tp = (data[high_col] + data[low_col] + data[close_col]) / 3
        
        # Calculate SMA of Typical Price
        sma_tp = tp.rolling(window=period).mean()
        
        # Calculate Mean Deviation
        mean_deviation = pd.Series(index=tp.index)
        
        for i in range(period - 1, len(tp)):
            mean_deviation.iloc[i] = abs(tp.iloc[i-period+1:i+1] - sma_tp.iloc[i]).mean()
            
        # Calculate CCI
        cci = (tp - sma_tp) / (0.015 * mean_deviation)
        
        return cci
