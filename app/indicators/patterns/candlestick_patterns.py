"""
Candlestick pattern recognition module for options recommendation platform.
Implements detection of various candlestick patterns for market analysis.
"""
import pandas as pd
import numpy as np

class CandlestickPatterns:
    """
    Class to detect candlestick patterns in price data
    """
    
    @staticmethod
    def detect_hammer(data, idx, body_pct_threshold=0.3, lower_wick_multiplier=2.0, upper_wick_pct_threshold=0.1):
        """
        Detect hammer candlestick pattern
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            body_pct_threshold (float): Maximum body size as percentage of total range
            lower_wick_multiplier (float): Minimum lower wick size as multiplier of body size
            upper_wick_pct_threshold (float): Maximum upper wick size as percentage of total range
            
        Returns:
            bool: True if hammer pattern detected, False otherwise
        """
        if idx < 0 or idx >= len(data):
            return False
        
        row = data.iloc[idx]
        
        # Calculate body and wick sizes
        body_size = abs(row['open'] - row['close'])
        total_range = row['high'] - row['low']
        
        if total_range == 0:  # Avoid division by zero
            return False
            
        lower_wick = min(row['open'], row['close']) - row['low']
        upper_wick = row['high'] - max(row['open'], row['close'])
        
        body_pct = body_size / total_range
        upper_wick_pct = upper_wick / total_range
        
        # Hammer criteria:
        # 1. Small body (less than threshold % of total range)
        # 2. Long lower wick (at least multiplier times the body size)
        # 3. Small or no upper wick (less than threshold % of total range)
        # 4. Body is in the upper portion of the candle
        is_hammer = (
            body_pct <= body_pct_threshold and 
            lower_wick >= lower_wick_multiplier * body_size and 
            upper_wick_pct <= upper_wick_pct_threshold and
            max(row['open'], row['close']) >= row['high'] - 0.3 * total_range
        )
        
        return is_hammer
    
    @staticmethod
    def detect_hanging_man(data, idx, body_pct_threshold=0.3, lower_wick_multiplier=2.0, upper_wick_pct_threshold=0.1):
        """
        Detect hanging man candlestick pattern (same shape as hammer but appears after an uptrend)
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            body_pct_threshold (float): Maximum body size as percentage of total range
            lower_wick_multiplier (float): Minimum lower wick size as multiplier of body size
            upper_wick_pct_threshold (float): Maximum upper wick size as percentage of total range
            
        Returns:
            bool: True if hanging man pattern detected, False otherwise
        """
        # First check if the candle has hammer shape
        if not CandlestickPatterns.detect_hammer(data, idx, body_pct_threshold, lower_wick_multiplier, upper_wick_pct_threshold):
            return False
        
        # Then check if it appears after an uptrend (at least 3 out of 5 previous candles are up)
        if idx < 5:
            return False
            
        up_candles = 0
        for i in range(idx-5, idx):
            if i >= 0 and data.iloc[i]['close'] > data.iloc[i]['open']:
                up_candles += 1
                
        return up_candles >= 3
    
    @staticmethod
    def detect_doji(data, idx, doji_threshold=0.05):
        """
        Detect doji candlestick pattern (open and close are very close)
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            doji_threshold (float): Maximum body size as percentage of total range to be considered a doji
            
        Returns:
            bool: True if doji pattern detected, False otherwise
        """
        if idx < 0 or idx >= len(data):
            return False
        
        row = data.iloc[idx]
        
        # Calculate body and total range
        body_size = abs(row['open'] - row['close'])
        total_range = row['high'] - row['low']
        
        if total_range == 0:  # Avoid division by zero
            return False
            
        body_pct = body_size / total_range
        
        # Doji criteria: very small body compared to total range
        return body_pct <= doji_threshold
    
    @staticmethod
    def detect_bullish_engulfing(data, idx):
        """
        Detect bullish engulfing candlestick pattern
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            
        Returns:
            bool: True if bullish engulfing pattern detected, False otherwise
        """
        if idx < 1 or idx >= len(data):
            return False
        
        current = data.iloc[idx]
        previous = data.iloc[idx-1]
        
        # Bullish engulfing criteria:
        # 1. Current candle is bullish (close > open)
        # 2. Previous candle is bearish (close < open)
        # 3. Current candle's body completely engulfs previous candle's body
        is_bullish_engulfing = (
            current['close'] > current['open'] and  # Current is bullish
            previous['close'] < previous['open'] and  # Previous is bearish
            current['open'] <= previous['close'] and  # Current open below or equal to previous close
            current['close'] >= previous['open']  # Current close above or equal to previous open
        )
        
        return is_bullish_engulfing
    
    @staticmethod
    def detect_bearish_engulfing(data, idx):
        """
        Detect bearish engulfing candlestick pattern
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            
        Returns:
            bool: True if bearish engulfing pattern detected, False otherwise
        """
        if idx < 1 or idx >= len(data):
            return False
        
        current = data.iloc[idx]
        previous = data.iloc[idx-1]
        
        # Bearish engulfing criteria:
        # 1. Current candle is bearish (close < open)
        # 2. Previous candle is bullish (close > open)
        # 3. Current candle's body completely engulfs previous candle's body
        is_bearish_engulfing = (
            current['close'] < current['open'] and  # Current is bearish
            previous['close'] > previous['open'] and  # Previous is bullish
            current['open'] >= previous['close'] and  # Current open above or equal to previous close
            current['close'] <= previous['open']  # Current close below or equal to previous open
        )
        
        return is_bearish_engulfing
    
    @staticmethod
    def detect_shooting_star(data, idx, body_pct_threshold=0.3, upper_wick_multiplier=2.0, lower_wick_pct_threshold=0.1):
        """
        Detect shooting star candlestick pattern
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            body_pct_threshold (float): Maximum body size as percentage of total range
            upper_wick_multiplier (float): Minimum upper wick size as multiplier of body size
            lower_wick_pct_threshold (float): Maximum lower wick size as percentage of total range
            
        Returns:
            bool: True if shooting star pattern detected, False otherwise
        """
        if idx < 0 or idx >= len(data):
            return False
        
        row = data.iloc[idx]
        
        # Calculate body and wick sizes
        body_size = abs(row['open'] - row['close'])
        total_range = row['high'] - row['low']
        
        if total_range == 0:  # Avoid division by zero
            return False
            
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        
        body_pct = body_size / total_range
        lower_wick_pct = lower_wick / total_range
        
        # Shooting star criteria:
        # 1. Small body (less than threshold % of total range)
        # 2. Long upper wick (at least multiplier times the body size)
        # 3. Small or no lower wick (less than threshold % of total range)
        # 4. Body is in the lower portion of the candle
        is_shooting_star = (
            body_pct <= body_pct_threshold and 
            upper_wick >= upper_wick_multiplier * body_size and 
            lower_wick_pct <= lower_wick_pct_threshold and
            min(row['open'], row['close']) <= row['low'] + 0.3 * total_range
        )
        
        # Check if it appears after an uptrend (at least 3 out of 5 previous candles are up)
        if is_shooting_star and idx >= 5:
            up_candles = 0
            for i in range(idx-5, idx):
                if data.iloc[i]['close'] > data.iloc[i]['open']:
                    up_candles += 1
                    
            return up_candles >= 3
        
        return False
    
    @staticmethod
    def detect_morning_star(data, idx, doji_threshold=0.05):
        """
        Detect morning star candlestick pattern (three-candle bullish reversal)
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            doji_threshold (float): Maximum body size as percentage of total range to be considered a doji
            
        Returns:
            bool: True if morning star pattern detected, False otherwise
        """
        if idx < 2 or idx >= len(data):
            return False
        
        first = data.iloc[idx-2]  # First candle (bearish)
        second = data.iloc[idx-1]  # Second candle (small/doji)
        third = data.iloc[idx]  # Third candle (bullish)
        
        # Calculate body sizes
        first_body = first['open'] - first['close']  # Positive for bearish
        second_body = abs(second['open'] - second['close'])
        third_body = third['close'] - third['open']  # Positive for bullish
        
        second_total_range = second['high'] - second['low']
        if second_total_range == 0:  # Avoid division by zero
            return False
            
        second_body_pct = second_body / second_total_range
        
        # Morning star criteria:
        # 1. First candle is bearish (close < open)
        # 2. Second candle has a small body (doji or near-doji)
        # 3. Third candle is bullish (close > open)
        # 4. Second candle's close is below first candle's close
        # 5. Third candle closes into first candle's body
        is_morning_star = (
            first_body > 0 and  # First is bearish
            second_body_pct <= doji_threshold and  # Second is doji-like
            third_body > 0 and  # Third is bullish
            max(second['open'], second['close']) < first['close'] and  # Gap down or lower close
            third['close'] > (first['open'] + first['close']) / 2  # Third closes into first body
        )
        
        return is_morning_star
    
    @staticmethod
    def detect_evening_star(data, idx, doji_threshold=0.05):
        """
        Detect evening star candlestick pattern (three-candle bearish reversal)
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            doji_threshold (float): Maximum body size as percentage of total range to be considered a doji
            
        Returns:
            bool: True if evening star pattern detected, False otherwise
        """
        if idx < 2 or idx >= len(data):
            return False
        
        first = data.iloc[idx-2]  # First candle (bullish)
        second = data.iloc[idx-1]  # Second candle (small/doji)
        third = data.iloc[idx]  # Third candle (bearish)
        
        # Calculate body sizes
        first_body = first['close'] - first['open']  # Positive for bullish
        second_body = abs(second['open'] - second['close'])
        third_body = third['open'] - third['close']  # Positive for bearish
        
        second_total_range = second['high'] - second['low']
        if second_total_range == 0:  # Avoid division by zero
            return False
            
        second_body_pct = second_body / second_total_range
        
        # Evening star criteria:
        # 1. First candle is bullish (close > open)
        # 2. Second candle has a small body (doji or near-doji)
        # 3. Third candle is bearish (close < open)
        # 4. Second candle's close is above first candle's close
        # 5. Third candle closes into first candle's body
        is_evening_star = (
            first_body > 0 and  # First is bullish
            second_body_pct <= doji_threshold and  # Second is doji-like
            third_body > 0 and  # Third is bearish
            min(second['open'], second['close']) > first['close'] and  # Gap up or higher close
            third['close'] < (first['open'] + first['close']) / 2  # Third closes into first body
        )
        
        return is_evening_star
    
    @staticmethod
    def detect_order_block(data, idx, num_candles_to_check=10, price_move_threshold=0.01):
        """
        Detect order block (institutional supply/demand zone before strong impulse move)
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to check for pattern
            num_candles_to_check (int): Number of candles to check after potential order block
            price_move_threshold (float): Minimum price move percentage to consider as strong impulse
            
        Returns:
            tuple: (is_bullish_ob, is_bearish_ob) - boolean flags for bullish and bearish order blocks
        """
        if idx < 1 or idx + num_candles_to_check >= len(data):
            return False, False
        
        current = data.iloc[idx]
        next_candle = data.iloc[idx+1]
        
        # Check for bullish order block (last red candle before strong bullish move)
        is_bullish_ob = False
        if current['close'] < current['open']:  # Current is bearish
            # Check if followed by strong bullish move
            start_price = next_candle['open']
            max_price = start_price
            
            for i in range(idx+1, idx+num_candles_to_check+1):
                candle = data.iloc[i]
                max_price = max(max_price, candle['high'])
            
            price_move_pct = (max_price - start_price) / start_price
            is_bullish_ob = price_move_pct >= price_move_threshold
        
        # Check for bearish order block (last green candle before strong bearish move)
        is_bearish_ob = False
        if current['close'] > current['open']:  # Current is bullish
            # Check if followed by strong bearish move
            start_price = next_candle['open']
            min_price = start_price
            
            for i in range(idx+1, idx+num_candles_to_check+1):
                candle = data.iloc[i]
                min_price = min(min_price, candle['low'])
            
            price_move_pct = (start_price - min_price) / start_price
            is_bearish_ob = price_move_pct >= price_move_threshold
        
        return is_bullish_ob, is_bearish_ob
    
    @staticmethod
    def analyze_candle(data, idx):
        """
        Analyze a single candle and detect all possible patterns
        
        Args:
            data (pd.DataFrame): OHLC price data
            idx (int): Index to analyze
            
        Returns:
            dict: Dictionary with detected patterns and their strengths
        """
        if idx < 0 or idx >= len(data):
            return {}
        
        patterns = {}
        
        # Check for single-candle patterns
        if CandlestickPatterns.detect_hammer(data, idx):
            patterns['hammer'] = 0.7  # Strength score
        
        if CandlestickPatterns.detect_hanging_man(data, idx):
            patterns['hanging_man'] = 0.6  # Strength score
        
        if CandlestickPatterns.detect_doji(data, idx):
            patterns['doji'] = 0.4  # Strength score
        
        if CandlestickPatterns.detect_shooting_star(data, idx):
            patterns['shooting_star'] = 0.65  # Strength score
        
        # Check for multi-candle patterns
        if CandlestickPatterns.detect_bullish_engulfing(data, idx):
            patterns['bullish_engulfing'] = 0.75  # Strength score
        
        if CandlestickPatterns.detect_bearish_engulfing(data, idx):
            patterns['bearish_engulfing'] = 0.75  # Strength score
        
        if CandlestickPatterns.detect_morning_star(data, idx):
            patterns['morning_star'] = 0.85  # Strength score
        
        if CandlestickPatterns.detect_evening_star(data, idx):
            patterns['evening_star'] = 0.85  # Strength score
        
        # Check for order blocks
        bullish_ob, bearish_ob = CandlestickPatterns.detect_order_block(data, idx)
        if bullish_ob:
            patterns['bullish_order_block'] = 0.8  # Strength score
        if bearish_ob:
            patterns['bearish_order_block'] = 0.8  # Strength score
        
        return patterns
    
    @staticmethod
    def analyze_candlestick_patterns(data, lookback=10):
        """
        Analyze price data and detect candlestick patterns
        
        Args:
            data (pd.DataFrame): OHLC price data
            lookback (int): Number of recent candles to analyze
            
        Returns:
            list: List of dictionaries with detected patterns and their details
        """
        if data.empty or len(data) < 3:  # Need at least 3 candles for patterns
            return []
        
        results = []
        start_idx = max(0, len(data) - lookback)
        
        for idx in range(start_idx, len(data)):
            patterns = CandlestickPatterns.analyze_candle(data, idx)
            
            if patterns:
                # Determine overall sentiment from patterns
                bullish_patterns = ['hammer', 'bullish_engulfing', 'morning_star', 'bullish_order_block']
                bearish_patterns = ['hanging_man', 'shooting_star', 'bearish_engulfing', 'evening_star', 'bearish_order_block']
                
                bullish_score = sum(patterns[p] for p in patterns if p in bullish_patterns)
                bearish_score = sum(patterns[p] for p in patterns if p in bearish_patterns)
                
                sentiment = 'neutral'
                if bullish_score > bearish_score:
                    sentiment = 'bullish'
                elif bearish_score > bullish_score:
                    sentiment = 'bearish'
                
                # Add to results
                results.append({
                    'index': idx,
                    'date': data.index[idx] if hasattr(data, 'index') else idx,
                    'patterns': patterns,
                    'sentiment': sentiment,
                    'strength': max(bullish_score, bearish_score)
                })
        
        return results
