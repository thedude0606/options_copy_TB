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
                    signals['signal_details']['fvg'] = f"Neutral (Bullish FVGs: {len(bullish_fvgs)}, Bearish FVGs: {len(bearish_fvgs)})"
        
        # Moving Averages signals
        if 'moving_averages' in indicators:
            ma_dict = indicators['moving_averages']
            if 'SMA_20' in ma_dict and 'SMA_50' in ma_dict and not ma_dict['SMA_20'].empty and not ma_dict['SMA_50'].empty:
                sma_20 = ma_dict['SMA_20'].iloc[-1]
                sma_50 = ma_dict['SMA_50'].iloc[-1]
                
                if sma_20 > sma_50:
                    signals['bullish'] += 1
                    signals['signal_details']['moving_averages'] = f"Bullish (SMA20: {sma_20:.2f} > SMA50: {sma_50:.2f})"
                elif sma_20 < sma_50:
                    signals['bearish'] += 1
                    signals['signal_details']['moving_averages'] = f"Bearish (SMA20: {sma_20:.2f} < SMA50: {sma_50:.2f})"
                else:
                    signals['neutral'] += 1
                    signals['signal_details']['moving_averages'] = f"Neutral (SMA20: {sma_20:.2f}, SMA50: {sma_50:.2f})"
        
        # Calculate overall signal
        total_signals = signals['bullish'] + signals['bearish'] + signals['neutral']
        if total_signals > 0:
            signals['bullish_pct'] = signals['bullish'] / total_signals
            signals['bearish_pct'] = signals['bearish'] / total_signals
            signals['neutral_pct'] = signals['neutral'] / total_signals
            
            if signals['bullish_pct'] > 0.5:
                signals['overall'] = 'bullish'
                signals['confidence'] = signals['bullish_pct']
            elif signals['bearish_pct'] > 0.5:
                signals['overall'] = 'bearish'
                signals['confidence'] = signals['bearish_pct']
            else:
                signals['overall'] = 'neutral'
                signals['confidence'] = signals['neutral_pct']
        else:
            signals['overall'] = 'neutral'
            signals['confidence'] = 0
        
        return signals
    
    def _score_options(self, options_data, signals, confidence_threshold):
        """
        Score options based on signals and options analysis
        
        Args:
            options_data (pd.DataFrame): Analyzed options data
            signals (dict): Dictionary of trading signals
            confidence_threshold (float): Minimum confidence score for recommendations
            
        Returns:
            pd.DataFrame: Scored options with recommendations
        """
        if options_data.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        recommendations = options_data.copy()
        
        # Filter based on overall signal
        overall_signal = signals['overall']
        confidence = signals['confidence']
        
        if confidence < confidence_threshold:
            # Not enough confidence for a recommendation
            return pd.DataFrame()
        
        # Filter options based on signal
        if overall_signal == 'bullish':
            # For bullish signal, recommend calls
            filtered_options = recommendations[recommendations['optionType'].str.lower() == 'call']
        elif overall_signal == 'bearish':
            # For bearish signal, recommend puts
            filtered_options = recommendations[recommendations['optionType'].str.lower() == 'put']
        else:
            # For neutral signal, no clear recommendation
            return pd.DataFrame()
        
        if filtered_options.empty:
            return pd.DataFrame()
        
        # Score options based on multiple factors
        for idx, row in filtered_options.iterrows():
            score = 0
            
            # Score based on probability of profit
            if 'probabilityOfProfit' in row and not pd.isna(row['probabilityOfProfit']):
                score += row['probabilityOfProfit'] * 40  # 40% weight
            
            # Score based on risk-reward ratio
            if 'riskRewardRatio' in row and not pd.isna(row['riskRewardRatio']):
                # Cap risk-reward at 5 for scoring
                capped_rr = min(row['riskRewardRatio'], 5)
                score += (capped_rr / 5) * 30  # 30% weight
            
            # Score based on delta (for directional alignment)
            if 'delta' in row and not pd.isna(row['delta']):
                if overall_signal == 'bullish':
                    # For calls, prefer delta around 0.5-0.7
                    delta_score = 1 - abs(row['delta'] - 0.6)
                else:
                    # For puts, prefer delta around -0.5 to -0.7
                    delta_score = 1 - abs(row['delta'] + 0.6)
                
                score += delta_score * 20  # 20% weight
            
            # Score based on days to expiration
            if 'daysToExpiration' in row and not pd.isna(row['daysToExpiration']):
                # Prefer 30-45 DTE
                days = row['daysToExpiration'].days
                if 20 <= days <= 60:
                    dte_score = 1 - abs(days - 40) / 40
                else:
                    dte_score = 0.2  # Lower score for very short or long DTE
                
                score += dte_score * 10  # 10% weight
            
            # Store the score
            filtered_options.loc[idx, 'score'] = score
        
        # Sort by score and take top recommendations
        top_recommendations = filtered_options.sort_values('score', ascending=False).head(5)
        
        # Add recommendation details
        for idx, row in top_recommendations.iterrows():
            # Format recommendation details
            details = {
                'type': row['optionType'],
                'strike': row['strikePrice'],
                'expiration': row['expirationDate'],
                'current_price': row['last'] if 'last' in row else row['mark'] if 'mark' in row else 0,
                'underlying_price': row['underlyingPrice'] if 'underlyingPrice' in row else None,
                'probability_of_profit': row['probabilityOfProfit'] if 'probabilityOfProfit' in row else None,
                'risk_reward_ratio': row['riskRewardRatio'] if 'riskRewardRatio' in row else None,
                'potential_return_pct': row['potentialReturnPercent'] if 'potentialReturnPercent' in row else None,
                'delta': row['delta'] if 'delta' in row else None,
                'gamma': row['gamma'] if 'gamma' in row else None,
                'theta': row['theta'] if 'theta' in row else None,
                'vega': row['vega'] if 'vega' in row else None,
                'confidence_score': confidence,
                'signal_details': signals['signal_details'],
                'overall_score': row['score']
            }
            
            top_recommendations.loc[idx, 'recommendation_details'] = str(details)
        
        return top_recommendations
