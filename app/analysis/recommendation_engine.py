"""
Recommendation engine module for options recommendation platform.
Implements recommendation logic based on technical indicators and options analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.indicators.technical_indicators import TechnicalIndicators
from app.analysis.options_analysis import OptionsAnalysis

class RecommendationEngine:
    """
    Class to generate options trading recommendations based on technical indicators and options analysis
    """
    
    def __init__(self, data_collector):
        """
        Initialize the recommendation engine
        
        Args:
            data_collector: DataCollector instance for retrieving market data
        """
        self.data_collector = data_collector
        self.technical_indicators = TechnicalIndicators()
        self.options_analysis = OptionsAnalysis()
    
    def generate_recommendations(self, symbol, lookback_days=30, confidence_threshold=0.6):
        """
        Generate options trading recommendations for a symbol
        
        Args:
            symbol (str): The stock symbol
            lookback_days (int): Number of days to look back for historical data
            confidence_threshold (float): Minimum confidence score for recommendations
            
        Returns:
            pd.DataFrame: Recommendations with details
        """
        try:
            # Get historical data
            historical_data = self.data_collector.get_historical_data(
                symbol=symbol,
                period_type='month',
                period=1,
                frequency_type='daily',
                frequency=1
            )
            
            if historical_data.empty:
                print(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Get options data
            options_data = self.data_collector.get_option_data(symbol)
            
            if options_data.empty:
                print(f"No options data available for {symbol}")
                return pd.DataFrame()
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(historical_data)
            
            # Calculate options Greeks and probabilities
            options_analysis = self._analyze_options(options_data)
            
            # Generate signals based on technical indicators
            signals = self._generate_signals(indicators)
            
            # Score options based on signals and options analysis
            recommendations = self._score_options(options_analysis, signals, confidence_threshold)
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    
    def _calculate_indicators(self, historical_data):
        """
        Calculate technical indicators for historical data
        
        Args:
            historical_data (pd.DataFrame): Historical price data
            
        Returns:
            dict: Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate RSI
        indicators['rsi'] = self.technical_indicators.calculate_rsi(historical_data)
        
        # Calculate MACD
        macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(historical_data)
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram
        
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = self.technical_indicators.calculate_bollinger_bands(historical_data)
        indicators['bollinger_middle'] = middle_band
        indicators['bollinger_upper'] = upper_band
        indicators['bollinger_lower'] = lower_band
        
        # Calculate IMI
        indicators['imi'] = self.technical_indicators.calculate_imi(historical_data)
        
        # Calculate MFI
        indicators['mfi'] = self.technical_indicators.calculate_mfi(historical_data)
        
        # Calculate Fair Value Gap
        indicators['fvg'] = self.technical_indicators.calculate_fair_value_gap(historical_data)
        
        # Calculate Liquidity Zones
        indicators['liquidity_zones'] = self.technical_indicators.calculate_liquidity_zones(historical_data)
        
        # Calculate Moving Averages
        indicators['moving_averages'] = self.technical_indicators.calculate_moving_averages(historical_data)
        
        # Calculate Volatility
        indicators['volatility'] = self.technical_indicators.calculate_volatility(historical_data)
        
        return indicators
    
    def _analyze_options(self, options_data):
        """
        Analyze options data with Greeks and probabilities
        
        Args:
            options_data (pd.DataFrame): Options data
            
        Returns:
            pd.DataFrame: Analyzed options data
        """
        # Calculate Greeks
        options_with_greeks = self.options_analysis.calculate_all_greeks(options_data)
        
        # Calculate probability of profit
        options_with_prob = self.options_analysis.calculate_probability_of_profit(options_with_greeks)
        
        # Calculate risk-reward ratio
        analyzed_options = self.options_analysis.calculate_risk_reward_ratio(options_with_prob)
        
        return analyzed_options
    
    def _generate_signals(self, indicators):
        """
        Generate trading signals based on technical indicators
        
        Args:
            indicators (dict): Dictionary of calculated indicators
            
        Returns:
            dict: Dictionary of trading signals
        """
        signals = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0,
            'signal_details': {}
        }
        
        # RSI signals
        if 'rsi' in indicators and not indicators['rsi'].empty:
            rsi = indicators['rsi'].iloc[-1]
            if rsi < 30:
                signals['bullish'] += 1
                signals['signal_details']['rsi'] = f"Bullish (RSI: {rsi:.2f} < 30)"
            elif rsi > 70:
                signals['bearish'] += 1
                signals['signal_details']['rsi'] = f"Bearish (RSI: {rsi:.2f} > 70)"
            else:
                signals['neutral'] += 1
                signals['signal_details']['rsi'] = f"Neutral (RSI: {rsi:.2f})"
        
        # MACD signals
        if all(k in indicators for k in ['macd_line', 'macd_signal', 'macd_histogram']):
            if not indicators['macd_line'].empty and not indicators['macd_signal'].empty:
                macd = indicators['macd_line'].iloc[-1]
                signal = indicators['macd_signal'].iloc[-1]
                histogram = indicators['macd_histogram'].iloc[-1]
                
                if macd > signal and histogram > 0:
                    signals['bullish'] += 1
                    signals['signal_details']['macd'] = f"Bullish (MACD: {macd:.2f} > Signal: {signal:.2f})"
                elif macd < signal and histogram < 0:
                    signals['bearish'] += 1
                    signals['signal_details']['macd'] = f"Bearish (MACD: {macd:.2f} < Signal: {signal:.2f})"
                else:
                    signals['neutral'] += 1
                    signals['signal_details']['macd'] = f"Neutral (MACD: {macd:.2f}, Signal: {signal:.2f})"
        
        # Bollinger Bands signals
        if all(k in indicators for k in ['bollinger_middle', 'bollinger_upper', 'bollinger_lower']):
            if not indicators['bollinger_middle'].empty:
                close = indicators['bollinger_middle'].index[-1]  # Assuming close price is the index
                upper = indicators['bollinger_upper'].iloc[-1]
                lower = indicators['bollinger_lower'].iloc[-1]
                middle = indicators['bollinger_middle'].iloc[-1]
                
                if close < lower:
                    signals['bullish'] += 1
                    signals['signal_details']['bollinger'] = f"Bullish (Price: {close:.2f} < Lower: {lower:.2f})"
                elif close > upper:
                    signals['bearish'] += 1
                    signals['signal_details']['bollinger'] = f"Bearish (Price: {close:.2f} > Upper: {upper:.2f})"
                else:
                    signals['neutral'] += 1
                    signals['signal_details']['bollinger'] = f"Neutral (Price: {close:.2f}, Middle: {middle:.2f})"
        
        # IMI signals
        if 'imi' in indicators and not indicators['imi'].empty:
            imi = indicators['imi'].iloc[-1]
            if imi > 70:
                signals['bullish'] += 1
                signals['signal_details']['imi'] = f"Bullish (IMI: {imi:.2f} > 70)"
            elif imi < 30:
                signals['bearish'] += 1
                signals['signal_details']['imi'] = f"Bearish (IMI: {imi:.2f} < 30)"
            else:
                signals['neutral'] += 1
                signals['signal_details']['imi'] = f"Neutral (IMI: {imi:.2f})"
        
        # MFI signals
        if 'mfi' in indicators and not indicators['mfi'].empty:
            mfi = indicators['mfi'].iloc[-1]
            if mfi < 20:
                signals['bullish'] += 1
                signals['signal_details']['mfi'] = f"Bullish (MFI: {mfi:.2f} < 20)"
            elif mfi > 80:
                signals['bearish'] += 1
                signals['signal_details']['mfi'] = f"Bearish (MFI: {mfi:.2f} > 80)"
            else:
                signals['neutral'] += 1
                signals['signal_details']['mfi'] = f"Neutral (MFI: {mfi:.2f})"
        
        # FVG signals
        if 'fvg' in indicators and not indicators['fvg'].empty:
            recent_fvgs = indicators['fvg'][indicators['fvg']['datetime'] >= (datetime.now() - timedelta(days=5))]
            
            if not recent_fvgs.empty:
                bullish_fvgs = recent_fvgs[recent_fvgs['type'] == 'bullish']
                bearish_fvgs = recent_fvgs[recent_fvgs['type'] == 'bearish']
                
                if len(bullish_fvgs) > len(bearish_fvgs):
                    signals['bullish'] += 1
                    signals['signal_details']['fvg'] = f"Bullish (Bullish FVGs: {len(bullish_fvgs)}, Bearish FVGs: {len(bearish_fvgs)})"
                elif len(bearish_fvgs) > len(bullish_fvgs):
                    signals['bearish'] += 1
                    signals['signal_details']['fvg'] = f"Bearish (Bullish FVGs: {len(bullish_fvgs)}, Bearish FVGs: {len(bearish_fvgs)})"
                else:
                    signals['neutral'] += 1
                    signals['signal_details']['fvg'] = f"Neutral (Equal FVGs)"
        
        # Liquidity Zones signals
        if 'liquidity_zones' in indicators and not indicators['liquidity_zones'].empty:
            zones = indicators['liquidity_zones']
            current_price = zones['price'].iloc[-1] if 'price' in zones.columns else None
            
            if current_price:
                support_zones = zones[zones['type'] == 'support']
                resistance_zones = zones[zones['type'] == 'resistance']
                
                if not support_zones.empty and not resistance_zones.empty:
                    nearest_support = support_zones.iloc[0]['level']
                    nearest_resistance = resistance_zones.iloc[0]['level']
                    
                    support_distance = abs(current_price - nearest_support) / current_price
                    resistance_distance = abs(nearest_resistance - current_price) / current_price
                    
                    if support_distance < resistance_distance:
                        signals['bullish'] += 1
                        signals['signal_details']['liquidity'] = f"Bullish (Closer to support: {nearest_support:.2f})"
                    else:
                        signals['bearish'] += 1
                        signals['signal_details']['liquidity'] = f"Bearish (Closer to resistance: {nearest_resistance:.2f})"
        
        # Moving Averages signals
        if 'moving_averages' in indicators:
            ma_data = indicators['moving_averages']
            if not ma_data.empty and 'ma_20' in ma_data.columns and 'ma_50' in ma_data.columns:
                ma_20 = ma_data['ma_20'].iloc[-1]
                ma_50 = ma_data['ma_50'].iloc[-1]
                
                if ma_20 > ma_50:
                    signals['bullish'] += 1
                    signals['signal_details']['moving_averages'] = f"Bullish (MA20: {ma_20:.2f} > MA50: {ma_50:.2f})"
                else:
                    signals['bearish'] += 1
                    signals['signal_details']['moving_averages'] = f"Bearish (MA20: {ma_20:.2f} < MA50: {ma_50:.2f})"
        
        # Volatility signals
        if 'volatility' in indicators and not indicators['volatility'].empty:
            volatility = indicators['volatility'].iloc[-1]
            historical_vol = indicators['volatility'].mean()
            
            if volatility < historical_vol * 0.8:
                signals['neutral'] += 1
                signals['signal_details']['volatility'] = f"Low volatility ({volatility:.2f})"
            elif volatility > historical_vol * 1.2:
                # High volatility favors options sellers
                signals['bearish'] += 1
                signals['signal_details']['volatility'] = f"High volatility ({volatility:.2f})"
        
        return signals
    
    def _score_options(self, options_data, signals, confidence_threshold):
        """
        Score options based on signals and options analysis
        
        Args:
            options_data (pd.DataFrame): Analyzed options data
            signals (dict): Dictionary of trading signals
            confidence_threshold (float): Minimum confidence score for recommendations
            
        Returns:
            pd.DataFrame: Scored options recommendations
        """
        if options_data.empty:
            return pd.DataFrame()
        
        # Determine overall market direction
        bullish_signals = signals['bullish']
        bearish_signals = signals['bearish']
        neutral_signals = signals['neutral']
        
        total_signals = bullish_signals + bearish_signals + neutral_signals
        if total_signals == 0:
            return pd.DataFrame()
        
        bullish_score = bullish_signals / total_signals
        bearish_score = bearish_signals / total_signals
        neutral_score = neutral_signals / total_signals
        
        market_direction = 'neutral'
        if bullish_score > 0.5 and bullish_score > bearish_score:
            market_direction = 'bullish'
        elif bearish_score > 0.5 and bearish_score > bullish_score:
            market_direction = 'bearish'
        
        # Filter options based on market direction
        if market_direction == 'bullish':
            # For bullish market, recommend calls or put credit spreads
            filtered_options = options_data[options_data['optionType'] == 'CALL']
        elif market_direction == 'bearish':
            # For bearish market, recommend puts or call credit spreads
            filtered_options = options_data[options_data['optionType'] == 'PUT']
        else:
            # For neutral market, recommend iron condors or straddles
            filtered_options = options_data
        
        if filtered_options.empty:
            return pd.DataFrame()
        
        # Score each option
        scores = []
        for _, row in filtered_options.iterrows():
            score = 0
            
            # Base score from market direction
            if market_direction == 'bullish' and row['optionType'] == 'CALL':
                score += bullish_score * 30  # 30% weight
            elif market_direction == 'bearish' and row['optionType'] == 'PUT':
                score += bearish_score * 30  # 30% weight
            elif market_direction == 'neutral':
                score += neutral_score * 30  # 30% weight
            
            # Score based on probability of profit
            if 'probabilityOfProfit' in row and not pd.isna(row['probabilityOfProfit']):
                pop_score = row['probabilityOfProfit']
                score += pop_score * 30  # 30% weight
            
            # Score based on risk-reward ratio
            if 'riskRewardRatio' in row and not pd.isna(row['riskRewardRatio']):
                rr_ratio = row['riskRewardRatio']
                if rr_ratio > 0:
                    rr_score = min(rr_ratio / 3, 1)  # Cap at 1
                    score += rr_score * 20  # 20% weight
            
            # Score based on delta (prefer 0.3-0.7 range)
            if 'delta' in row and not pd.isna(row['delta']):
                delta = abs(row['delta'])
                if 0.3 <= delta <= 0.7:
                    delta_score = 1 - abs(delta - 0.5) / 0.5
                else:
                    delta_score = 1 - abs(row['delta'] + 0.6)
                
                score += delta_score * 20  # 20% weight
            
            # Score based on days to expiration
            if 'daysToExpiration' in row and not pd.isna(row['daysToExpiration']):
                # Prefer 30-45 DTE
                # Convert Timestamp to days if it's a Timestamp object
                if isinstance(row['daysToExpiration'], pd.Timedelta):
                    days = row['daysToExpiration'].days
                else:
                    # If it's already a number, use it directly
                    days = float(row['daysToExpiration'])
                
                if 20 <= days <= 60:
                    dte_score = 1 - abs(days - 40) / 40
                else:
                    dte_score = 0.2  # Lower score for very short or long DTE
                
                score += dte_score * 20  # 20% weight
            
            # Add to scores list
            scores.append({
                'symbol': row['symbol'] if 'symbol' in row else '',
                'optionType': row['optionType'],
                'strikePrice': row['strikePrice'] if 'strikePrice' in row else 0,
                'expirationDate': row['expirationDate'] if 'expirationDate' in row else '',
                'bid': row['bid'] if 'bid' in row else 0,
                'ask': row['ask'] if 'ask' in row else 0,
                'delta': row['delta'] if 'delta' in row else 0,
                'gamma': row['gamma'] if 'gamma' in row else 0,
                'theta': row['theta'] if 'theta' in row else 0,
                'vega': row['vega'] if 'vega' in row else 0,
                'probabilityOfProfit': row['probabilityOfProfit'] if 'probabilityOfProfit' in row else 0,
                'riskRewardRatio': row['riskRewardRatio'] if 'riskRewardRatio' in row else 0,
                'daysToExpiration': days if 'daysToExpiration' in row and not pd.isna(row['daysToExpiration']) else 0,
                'score': score,
                'confidence': score / 100,  # Convert to 0-1 scale
                'marketDirection': market_direction,
                'signalDetails': signals['signal_details']
            })
        
        # Convert to DataFrame and filter by confidence threshold
        recommendations_df = pd.DataFrame(scores)
        if not recommendations_df.empty:
            recommendations_df = recommendations_df[recommendations_df['confidence'] >= confidence_threshold]
            recommendations_df = recommendations_df.sort_values('confidence', ascending=False)
        
        return recommendations_df
    
    def get_underlying_price(self, symbol):
        """
        Get the current price of the underlying asset
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            float: Current price
        """
        try:
            quote = self.data_collector.get_quote(symbol)
            if quote and 'lastPrice' in quote:
                return quote['lastPrice']
            else:
                print("No underlying price available")
                return None
        except Exception as e:
            print(f"Error retrieving underlying price: {str(e)}")
            return None
