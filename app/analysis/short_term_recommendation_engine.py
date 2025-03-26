"""
Timeframe-specific recommendation engine for short-term options trading.
Optimized for 15, 30, 60, and 120-minute trading windows.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from app.indicators.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands

class ShortTermRecommendationEngine:
    """
    Recommendation engine optimized for short-term options trading timeframes
    """
    def __init__(self, data_pipeline):
        """
        Initialize the short-term recommendation engine
        
        Args:
            data_pipeline: ShortTermDataPipeline instance for retrieving market data
        """
        self.data_pipeline = data_pipeline
        self.logger = logging.getLogger('short_term_recommendation_engine')
        
        # Define timeframe-specific parameters
        self.timeframe_params = {
            '15m': {
                'rsi_period': 9,
                'rsi_overbought': 75,
                'rsi_oversold': 25,
                'macd_fast': 6,
                'macd_slow': 13,
                'macd_signal': 4,
                'bb_period': 10,
                'bb_std': 2.0,
                'min_volume_percentile': 70,
                'volatility_weight': 1.5,
                'momentum_weight': 2.0,
                'sentiment_weight': 0.8
            },
            '30m': {
                'rsi_period': 10,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 8,
                'macd_slow': 17,
                'macd_signal': 5,
                'bb_period': 14,
                'bb_std': 2.0,
                'min_volume_percentile': 65,
                'volatility_weight': 1.3,
                'momentum_weight': 1.8,
                'sentiment_weight': 1.0
            },
            '60m': {
                'rsi_period': 12,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 10,
                'macd_slow': 21,
                'macd_signal': 7,
                'bb_period': 20,
                'bb_std': 2.0,
                'min_volume_percentile': 60,
                'volatility_weight': 1.2,
                'momentum_weight': 1.5,
                'sentiment_weight': 1.2
            },
            '120m': {
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 26,
                'bb_std': 2.0,
                'min_volume_percentile': 55,
                'volatility_weight': 1.0,
                'momentum_weight': 1.2,
                'sentiment_weight': 1.5
            }
        }
    
    def generate_recommendations(self, symbol, timeframe='30m', option_type='ALL', 
                                min_confidence=0.6, min_risk_reward=1.5, 
                                indicator_weights=None):
        """
        Generate options recommendations for short-term trading
        
        Args:
            symbol (str): Stock symbol
            timeframe (str): Trading timeframe ('15m', '30m', '60m', '120m')
            option_type (str): Option type filter ('CALL', 'PUT', 'ALL')
            min_confidence (float): Minimum confidence score (0.0-1.0)
            min_risk_reward (float): Minimum risk/reward ratio
            indicator_weights (dict): Custom weights for technical indicators
            
        Returns:
            list: Recommendations sorted by confidence score
        """
        if timeframe not in self.timeframe_params:
            self.logger.warning(f"Invalid timeframe: {timeframe}, defaulting to 30m")
            timeframe = '30m'
            
        # Get price data for analysis
        price_data = self.data_pipeline.get_short_term_price_data(symbol, timeframe)
        if price_data.empty:
            self.logger.warning(f"No price data available for {symbol} {timeframe}")
            return []
            
        # Get options data
        options_data = self.data_pipeline.get_options_data_for_timeframe(symbol, timeframe)
        if not options_data or not options_data.get('options'):
            self.logger.warning(f"No options data available for {symbol}")
            return []
            
        # Get real-time data
        real_time_data = self.data_pipeline.get_real_time_data(symbol)
        if not real_time_data:
            self.logger.warning(f"No real-time data available for {symbol}")
            return []
            
        # Get sentiment data
        sentiment_data = self.data_pipeline.get_sentiment_data(symbol)
        
        # Apply technical analysis
        signals = self._analyze_technical_indicators(price_data, timeframe, indicator_weights)
        
        # Generate recommendations based on signals
        recommendations = self._generate_option_recommendations(
            symbol, 
            options_data, 
            signals, 
            real_time_data,
            sentiment_data,
            timeframe, 
            option_type, 
            min_confidence, 
            min_risk_reward
        )
        
        return recommendations
    
    def _analyze_technical_indicators(self, price_data, timeframe, custom_weights=None):
        """
        Analyze technical indicators for the given price data
        
        Args:
            price_data (pd.DataFrame): Price data with OHLCV columns
            timeframe (str): Trading timeframe
            custom_weights (dict): Custom weights for indicators
            
        Returns:
            dict: Technical analysis signals
        """
        # Get timeframe-specific parameters
        params = self.timeframe_params[timeframe]
        
        # Use custom weights if provided, otherwise use defaults
        weights = {
            'rsi': 5,
            'macd': 5,
            'bollinger': 5,
            'volume': 3,
            'trend': 4,
            'volatility': 3
        }
        
        if custom_weights:
            for key, value in custom_weights.items():
                if key in weights:
                    weights[key] = value
        
        # Calculate RSI
        rsi_data = calculate_rsi(
            price_data,
            period=params['rsi_period'],
            overbought=params['rsi_overbought'],
            oversold=params['rsi_oversold']
        )
        
        # Calculate MACD
        macd_data = calculate_macd(
            price_data,
            fast_period=params['macd_fast'],
            slow_period=params['macd_slow'],
            signal_period=params['macd_signal']
        )
        
        # Calculate Bollinger Bands
        bb_data = calculate_bollinger_bands(
            price_data,
            period=params['bb_period'],
            std_dev=params['bb_std']
        )
        
        # Calculate additional indicators
        price_data['returns'] = price_data['close'].pct_change()
        price_data['volatility'] = price_data['returns'].rolling(window=params['bb_period']).std()
        
        # Determine trend direction
        price_data['sma_short'] = price_data['close'].rolling(window=params['macd_fast']).mean()
        price_data['sma_long'] = price_data['close'].rolling(window=params['macd_slow']).mean()
        price_data['trend'] = np.where(price_data['sma_short'] > price_data['sma_long'], 1, -1)
        
        # Get the most recent data point
        latest = price_data.iloc[-1]
        
        # Determine signals
        signals = {
            'rsi': {
                'value': latest.get('rsi', 50),
                'signal': 'bullish' if latest.get('rsi', 50) < params['rsi_oversold'] else 
                          'bearish' if latest.get('rsi', 50) > params['rsi_overbought'] else 'neutral',
                'strength': abs((latest.get('rsi', 50) - 50) / 50),
                'weight': weights['rsi']
            },
            'macd': {
                'value': latest.get('macd', 0),
                'signal': 'bullish' if latest.get('macd', 0) > latest.get('macd_signal', 0) else 'bearish',
                'strength': abs(latest.get('macd', 0) - latest.get('macd_signal', 0)) / max(0.01, abs(latest.get('macd', 0))),
                'weight': weights['macd']
            },
            'bollinger': {
                'value': (latest.get('close', 0) - latest.get('bb_middle', 0)) / 
                         max(0.01, (latest.get('bb_upper', 0) - latest.get('bb_middle', 0))),
                'signal': 'bullish' if latest.get('close', 0) < latest.get('bb_lower', 0) else 
                          'bearish' if latest.get('close', 0) > latest.get('bb_upper', 0) else 'neutral',
                'strength': abs((latest.get('close', 0) - latest.get('bb_middle', 0)) / 
                               max(0.01, (latest.get('bb_upper', 0) - latest.get('bb_middle', 0)))),
                'weight': weights['bollinger']
            },
            'volume': {
                'value': latest.get('volume', 0),
                'signal': 'bullish' if latest.get('volume', 0) > price_data['volume'].quantile(params['min_volume_percentile']/100) else 'neutral',
                'strength': min(1.0, latest.get('volume', 0) / max(0.01, price_data['volume'].mean())),
                'weight': weights['volume']
            },
            'trend': {
                'value': latest.get('trend', 0),
                'signal': 'bullish' if latest.get('trend', 0) > 0 else 'bearish',
                'strength': min(1.0, abs(latest.get('sma_short', 0) - latest.get('sma_long', 0)) / 
                               max(0.01, latest.get('sma_long', 0))),
                'weight': weights['trend']
            },
            'volatility': {
                'value': latest.get('volatility', 0),
                'signal': 'neutral',
                'strength': min(1.0, latest.get('volatility', 0) / max(0.001, price_data['volatility'].mean())),
                'weight': weights['volatility']
            }
        }
        
        # Calculate overall market direction
        bullish_weight = 0
        bearish_weight = 0
        total_weight = 0
        
        for indicator, data in signals.items():
            if data['signal'] == 'bullish':
                bullish_weight += data['strength'] * data['weight']
            elif data['signal'] == 'bearish':
                bearish_weight += data['strength'] * data['weight']
            total_weight += data['weight']
        
        # Normalize to get a score between -1 (bearish) and 1 (bullish)
        market_direction = (bullish_weight - bearish_weight) / max(1, total_weight)
        
        signals['overall'] = {
            'market_direction': market_direction,
            'signal': 'bullish' if market_direction > 0.2 else 'bearish' if market_direction < -0.2 else 'neutral',
            'strength': abs(market_direction)
        }
        
        return signals
    
    def _generate_option_recommendations(self, symbol, options_data, signals, real_time_data, 
                                        sentiment_data, timeframe, option_type, 
                                        min_confidence, min_risk_reward):
        """
        Generate option recommendations based on technical signals
        
        Args:
            symbol (str): Stock symbol
            options_data (dict): Options chain data
            signals (dict): Technical analysis signals
            real_time_data (dict): Real-time market data
            sentiment_data (dict): Sentiment data
            timeframe (str): Trading timeframe
            option_type (str): Option type filter
            min_confidence (float): Minimum confidence score
            min_risk_reward (float): Minimum risk/reward ratio
            
        Returns:
            list: Filtered and sorted recommendations
        """
        # Get timeframe-specific parameters
        params = self.timeframe_params[timeframe]
        
        # Get current price
        current_price = real_time_data.get('last_price', options_data.get('underlying_price', 0))
        if current_price == 0:
            self.logger.warning(f"Invalid current price for {symbol}")
            return []
        
        # Determine which option types to consider based on signals and filters
        market_direction = signals['overall']['market_direction']
        market_signal = signals['overall']['signal']
        
        consider_calls = (option_type in ['ALL', 'CALL']) and (market_signal == 'bullish' or abs(market_direction) < 0.2)
        consider_puts = (option_type in ['ALL', 'PUT']) and (market_signal == 'bearish' or abs(market_direction) < 0.2)
        
        # Filter options based on expiration and type
        filtered_options = []
        for option in options_data.get('options', []):
            option_type_match = (
                (consider_calls and option.get('option_type') == 'CALL') or
                (consider_puts and option.get('option_type') == 'PUT')
            )
            
            if not option_type_match:
                continue
                
            # Add to filtered options
            filtered_options.append(option)
        
        # Calculate scores for each option
        recommendations = []
        for option in filtered_options:
            # Extract option data
            opt_type = option.get('option_type')
            strike = option.get('strike', 0)
            bid = option.get('bid', 0)
            ask = option.get('ask', 0)
            
            # Skip options with no liquidity
            if bid <= 0 or ask <= 0:
                continue
                
            # Calculate mid price
            mid_price = (bid + ask) / 2
            
            # Calculate moneyness (how far in/out of the money)
            if opt_type == 'CALL':
                moneyness = (current_price / strike) - 1
            else:  # PUT
                moneyness = 1 - (current_price / strike)
            
            # Calculate expected move based on timeframe
            timeframe_hours = {
                '15m': 0.25,
                '30m': 0.5,
                '60m': 1.0,
                '120m': 2.0
            }.get(timeframe, 0.5)
            
            # Use volatility to estimate expected move
            volatility = option.get('implied_volatility', 0.3)  # Default to 30% if not available
            annual_trading_days = 252
            annual_trading_hours = annual_trading_days * 6.5  # ~6.5 trading hours per day
            
            # Calculate expected move as percentage - use a more conservative approach
            # Square root of time rule from option pricing theory
            expected_move_pct = volatility * np.sqrt(timeframe_hours / annual_trading_hours) * 0.5  # Added dampening factor
            expected_move = current_price * expected_move_pct
            
            # Calculate target price based on expected move and direction
            direction_multiplier = 1 if opt_type == 'CALL' else -1
            signal_strength = signals['overall']['strength']
            
            # Adjust expected move based on signal strength - use a more conservative adjustment
            # Limit the impact of signal strength to avoid extreme moves
            signal_adjustment = max(-0.5, min(0.5, signal_strength))  # Limit to [-0.5, 0.5]
            adjusted_move = expected_move * (1 + signal_adjustment)
            
            # Calculate target price
            target_price = current_price + (direction_multiplier * adjusted_move)
            
            # Calculate potential profit/loss with time value consideration
            days_to_expiration = option.get('days_to_expiration', 30)
            time_value_factor = min(1.0, max(0.1, days_to_expiration / 30))  # Scale based on expiration
            
            if opt_type == 'CALL':
                intrinsic_value = max(0, target_price - strike)
                # Discount potential profit based on time to expiration
                potential_profit = (intrinsic_value * time_value_factor) - mid_price
            else:  # PUT
                intrinsic_value = max(0, strike - target_price)
                # Discount potential profit based on time to expiration
                potential_profit = (intrinsic_value * time_value_factor) - mid_price
            
            # Ensure potential profit is not negative
            potential_profit = max(0, potential_profit)
            
            # Calculate potential return with scaling based on expiration
            # Shorter-term options should have lower expected returns
            if mid_price > 0:
                base_return = potential_profit / mid_price
                expiration_scale = np.sqrt(time_value_factor)  # Square root scaling for more granularity
                potential_return = base_return * expiration_scale
            else:
                potential_return = 0
            
            # Apply reasonable upper bound to potential return (cap at 1000%)
            potential_return = min(10.0, potential_return)
            
            # Calculate maximum loss (simplified)
            max_loss = mid_price
            
            # Calculate risk/reward ratio with scaling based on option characteristics
            if max_loss > 0:
                base_risk_reward = potential_profit / max_loss
                
                # Scale risk/reward based on option characteristics
                # Options closer to expiration should have lower risk/reward
                # Options with higher implied volatility should have lower risk/reward
                volatility_factor = max(0.2, min(1.0, 0.5 / volatility))  # Lower for high volatility options
                expiration_factor = max(0.2, min(1.0, days_to_expiration / 60))  # Lower for near-term options
                
                risk_reward = base_risk_reward * volatility_factor * expiration_factor
            else:
                risk_reward = 0
            
            # Apply reasonable upper bound to risk/reward (cap at 100x)
            risk_reward = min(100.0, risk_reward)
            
            # Skip options with insufficient risk/reward
            if risk_reward < min_risk_reward:
                continue
            
            # Calculate confidence score with more granular approach
            # Base confidence on signal strength, moneyness, and liquidity
            signal_confidence = (signal_strength + 1) / 2  # Convert from [-1,1] to [0,1]
            
            # Prefer slightly OTM options for directional plays - more granular scoring
            # Highest score for options that are 2-5% OTM
            if opt_type == 'CALL':
                if moneyness < -0.05:  # Deep OTM
                    moneyness_score = max(0, 0.5 + (moneyness + 0.05) * 5)  # Ramp up from deep OTM
                elif moneyness < -0.02:  # Slightly OTM (sweet spot)
                    moneyness_score = 0.9
                elif moneyness < 0:  # Very slightly OTM
                    moneyness_score = 0.8
                elif moneyness < 0.05:  # Slightly ITM
                    moneyness_score = 0.7 - moneyness * 2
                else:  # Deep ITM
                    moneyness_score = max(0.1, 0.6 - moneyness)
            else:  # PUT
                if moneyness < -0.05:  # Deep ITM for puts
                    moneyness_score = max(0.1, 0.6 + moneyness)
                elif moneyness < 0:  # Slightly ITM for puts
                    moneyness_score = 0.7 + moneyness * 2
                elif moneyness < 0.02:  # Very slightly OTM
                    moneyness_score = 0.8
                elif moneyness < 0.05:  # Slightly OTM (sweet spot)
                    moneyness_score = 0.9
                else:  # Deep OTM
                    moneyness_score = max(0, 0.5 - (moneyness - 0.05) * 5)  # Ramp down for deep OTM
            
            # Liquidity score based on bid-ask spread - more granular
            spread_pct = (ask - bid) / mid_price if mid_price > 0 else 1
            if spread_pct < 0.02:  # Very tight spread
                liquidity_score = 1.0
            elif spread_pct < 0.05:  # Good spread
                liquidity_score = 0.9
            elif spread_pct < 0.1:  # Acceptable spread
                liquidity_score = 0.7
            elif spread_pct < 0.2:  # Wide spread
                liquidity_score = 0.5
            else:  # Very wide spread
                liquidity_score = max(0, 1 - spread_pct * 2)
            
            # Volume score with more granular approach
            volume = option.get('volume', 0)
            open_interest = option.get('open_interest', 0)
            combined_volume = volume + open_interest
            
            if combined_volume > 10000:
                volume_score = 1.0
            elif combined_volume > 5000:
                volume_score = 0.9
            elif combined_volume > 1000:
                volume_score = 0.8
            elif combined_volume > 500:
                volume_score = 0.7
            elif combined_volume > 100:
                volume_score = 0.5
            else:
                volume_score = max(0.1, combined_volume / 1000)
            
            # Sentiment adjustment with more granular approach
            sentiment_score = (sentiment_data.get('sentiment_score', 0) + 1) / 2  # Convert to [0,1]
            sentiment_adjustment = 1 + (sentiment_score - 0.5) * params['sentiment_weight'] * 0.5  # Reduced impact
            
            # Calculate final confidence score with weights
            # Use a more balanced weighting system
            confidence_score = (
                signal_confidence * 0.3 +
                moneyness_score * 0.25 +
                liquidity_score * 0.2 +
                volume_score * 0.15 +
                (risk_reward / 20) * 0.1  # Normalized risk/reward with reduced impact
            ) * sentiment_adjustment
            
            # Ensure confidence score is within reasonable bounds (0-1)
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            # Skip options with insufficient confidence
            if confidence_score < min_confidence:
                continue
            
            # Create recommendation object
            recommendation = {
                'symbol': symbol,
                'option_symbol': option.get('symbol', ''),
                'option_type': opt_type,
                'strike': strike,
                'expiration': option.get('expiration', ''),
                'bid': bid,
                'ask': ask,
                'entry_price': mid_price,
                'current_price': current_price,
                'target_price': target_price,
                'potential_profit': potential_profit,
                'potential_return': potential_return,
                'max_loss': max_loss,
                'risk_reward_ratio': risk_reward,
                'confidence_score': confidence_score,
                'volume': volume,
                'open_interest': open_interest,
                'implied_volatility': volatility,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'technical_signals': {
                    'market_direction': market_direction,
                    'market_signal': market_signal,
                    'signal_strength': signal_strength,
                    'rsi': signals['rsi']['value'],
                    'macd': signals['macd']['value'],
                    'bollinger': signals['bollinger']['value']
                },
                'sentiment': {
                    'score': sentiment_data.get('sentiment_score', 0),
                    'label': sentiment_data.get('sentiment_label', 'Neutral')
                }
            }
            
            recommendations.append(recommendation)
        
        # Sort recommendations by confidence score (descending)
        recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return recommendations
    
    def get_validation_data(self, recommendation, timeframe='30m'):
        """
        Get validation data for a specific recommendation
        
        Args:
            recommendation (dict): Recommendation data
            timeframe (str): Trading timeframe
            
        Returns:
            dict: Validation data for visualization
        """
        symbol = recommendation.get('symbol')
        option_type = recommendation.get('option_type')
        
        # Get price data for validation
        price_data = self.data_pipeline.get_short_term_price_data(symbol, timeframe)
        if price_data.empty:
            return {'error': 'No price data available'}
        
        # Get the most recent candles for visualization
        recent_candles = price_data.tail(20).copy()
        
        # Calculate key levels
        current_price = recommendation.get('current_price')
        strike_price = recommendation.get('strike')
        target_price = recommendation.get('target_price')
        
        # Calculate support and resistance levels
        price_range = recent_candles['high'].max() - recent_candles['low'].min()
        support_level = max(recent_candles['low'].min(), current_price - price_range * 0.5)
        resistance_level = min(recent_candles['high'].max(), current_price + price_range * 0.5)
        
        # Get technical indicators for validation
        rsi_data = calculate_rsi(price_data)
        macd_data = calculate_macd(price_data)
        bb_data = calculate_bollinger_bands(price_data)
        
        # Prepare validation data
        validation_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'option_type': option_type,
            'price_data': recent_candles.to_dict(orient='records'),
            'key_levels': {
                'current_price': current_price,
                'strike_price': strike_price,
                'target_price': target_price,
                'support_level': support_level,
                'resistance_level': resistance_level
            },
            'indicators': {
                'rsi': rsi_data.tail(20).to_dict(orient='records'),
                'macd': macd_data.tail(20).to_dict(orient='records'),
                'bollinger_bands': bb_data.tail(20).to_dict(orient='records')
            },
            'recommendation': recommendation
        }
        
        return validation_data
