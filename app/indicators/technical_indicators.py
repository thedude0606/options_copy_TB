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
        if data.empty or len(data) < period or not all(col in data.columns for col in ['open', 'close']):
            return pd.Series()
        
        # Calculate whether close is higher than open
        up_sum = ((data['close'] > data['open']).astype(int) * (data['close'] - data['open'])).rolling(window=period).sum()
        
        # Calculate total movement
        total_sum = (abs(data['close'] - data['open'])).rolling(window=period).sum()
        
        # Calculate IMI
        imi = 100 * up_sum / total_sum
        
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
        if data.empty or len(data) < period or not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            return pd.Series()
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * data['volume']
        
        # Calculate money flow direction
        positive_flow = (typical_price > typical_price.shift(1)).astype(int) * raw_money_flow
        negative_flow = (typical_price < typical_price.shift(1)).astype(int) * raw_money_flow
        
        # Calculate money flow ratio
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        # Handle division by zero
        money_flow_ratio = np.where(negative_sum != 0, positive_sum / negative_sum, 100)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return pd.Series(mfi, index=data.index)
    
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
            pd.Series: Volatility values (annualized)
        """
        if data.empty or len(data) < period + 1:
            return pd.Series()
        
        # Calculate log returns
        log_returns = np.log(data[price_col] / data[price_col].shift(1))
        
        # Calculate rolling standard deviation
        rolling_std = log_returns.rolling(window=period).std()
        
        # Annualize (assuming daily data, multiply by sqrt(252))
        annualized_vol = rolling_std * np.sqrt(252)
        
        return annualized_vol
