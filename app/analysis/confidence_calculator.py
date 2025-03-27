"""
Confidence probability calculation system for options recommendation platform.
Integrates technical analysis, profit prediction, and market conditions to provide confidence scores.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class ConfidenceCalculator:
    """
    Class to calculate confidence probabilities for options recommendations
    """
    
    def __init__(self, multi_timeframe_analyzer=None, profit_predictor=None):
        """
        Initialize the confidence calculator
        
        Args:
            multi_timeframe_analyzer: MultiTimeframeAnalyzer instance for technical analysis
            profit_predictor: ProfitPredictor instance for profit projections
        """
        self.logger = logging.getLogger(__name__)
        self.multi_timeframe_analyzer = multi_timeframe_analyzer
        self.profit_predictor = profit_predictor
        
        # Define weights for different factors in confidence calculation
        self.weights = {
            'technical_analysis': 0.35,  # Weight for technical indicators and patterns
            'profit_potential': 0.30,    # Weight for profit projection and probability
            'market_conditions': 0.20,   # Weight for overall market conditions
            'option_metrics': 0.15       # Weight for option-specific metrics (Greeks, etc.)
        }
        
        # Initialize factor scoring functions
        self.factor_scorers = {
            'technical_analysis': self._score_technical_analysis,
            'profit_potential': self._score_profit_potential,
            'market_conditions': self._score_market_conditions,
            'option_metrics': self._score_option_metrics
        }
    
    def calculate_confidence(self, symbol, option_data, market_data=None, risk_tolerance='moderate'):
        """
        Calculate confidence probability for an option recommendation
        
        Args:
            symbol (str): The underlying symbol
            option_data (dict): Option data including price, strike, expiration, etc.
            market_data (dict): Market data including indices, volatility, etc.
            risk_tolerance (str): Risk tolerance level ('conservative', 'moderate', 'aggressive')
            
        Returns:
            dict: Confidence calculation results
        """
        try:
            self.logger.info(f"Calculating confidence for {symbol} option: {option_data.get('optionType', '')} {option_data.get('strikePrice', '')}")
            
            # Initialize confidence data
            confidence_data = {
                'symbol': symbol,
                'option_type': option_data.get('optionType', ''),
                'strike_price': option_data.get('strikePrice', 0),
                'expiration_date': option_data.get('expirationDate', ''),
                'risk_tolerance': risk_tolerance,
                'calculation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'factors': {},
                'confidence_score': 0,
                'confidence_level': 'low',
                'signal_details': []
            }
            
            # Calculate scores for each factor
            factor_scores = {}
            for factor, weight in self.weights.items():
                scorer = self.factor_scorers.get(factor)
                if scorer:
                    score, details = scorer(symbol, option_data, market_data, risk_tolerance)
                    factor_scores[factor] = {
                        'score': score,
                        'weight': weight,
                        'weighted_score': score * weight,
                        'details': details
                    }
                    
                    # Add signal details
                    confidence_data['signal_details'].extend(details)
            
            # Store factor scores
            confidence_data['factors'] = factor_scores
            
            # Calculate overall confidence score (0-100)
            total_score = sum(factor['weighted_score'] for factor in factor_scores.values())
            confidence_data['confidence_score'] = total_score * 100  # Convert to percentage
            
            # Determine confidence level
            if total_score >= 0.8:
                confidence_data['confidence_level'] = 'very_high'
            elif total_score >= 0.65:
                confidence_data['confidence_level'] = 'high'
            elif total_score >= 0.5:
                confidence_data['confidence_level'] = 'moderate'
            elif total_score >= 0.35:
                confidence_data['confidence_level'] = 'low'
            else:
                confidence_data['confidence_level'] = 'very_low'
            
            # Add recommendation strength based on confidence and option type
            confidence_data['recommendation'] = self._generate_recommendation(
                confidence_data['confidence_level'], 
                option_data.get('optionType', ''),
                risk_tolerance
            )
            
            return confidence_data
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return {
                'symbol': symbol,
                'option_type': option_data.get('optionType', ''),
                'strike_price': option_data.get('strikePrice', 0),
                'error': str(e),
                'confidence_score': 0,
                'confidence_level': 'unknown'
            }
    
    def _score_technical_analysis(self, symbol, option_data, market_data, risk_tolerance):
        """
        Score technical analysis factors
        
        Args:
            symbol (str): The underlying symbol
            option_data (dict): Option data
            market_data (dict): Market data
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            tuple: (score, details)
        """
        score = 0.0
        details = []
        
        # Get option type (call or put)
        option_type = option_data.get('optionType', '').lower()
        is_call = option_type == 'call'
        
        # If we have multi-timeframe analyzer, use it
        if self.multi_timeframe_analyzer:
            try:
                # Analyze technical indicators across timeframes
                analysis = self.multi_timeframe_analyzer.analyze_multi_timeframe(symbol)
                
                if analysis and 'combined_signals' in analysis:
                    combined = analysis['combined_signals']
                    
                    # Get overall sentiment
                    sentiment = combined.get('overall_sentiment', 'neutral')
                    
                    # Calculate alignment with option type
                    if (is_call and sentiment == 'bullish') or (not is_call and sentiment == 'bearish'):
                        # Sentiment aligns with option type
                        alignment_score = combined.get('confidence', 0)
                        score += alignment_score
                        details.append(f"Technical sentiment ({sentiment}) aligns with {option_type} option: +{alignment_score:.2f}")
                    elif sentiment == 'neutral':
                        # Neutral sentiment
                        score += 0.3
                        details.append(f"Technical sentiment is neutral: +0.30")
                    else:
                        # Sentiment contradicts option type
                        alignment_score = 0.1
                        score += alignment_score
                        details.append(f"Technical sentiment ({sentiment}) contradicts {option_type} option: +{alignment_score:.2f}")
                    
                    # Add details from signal details
                    for signal in combined.get('signal_details', []):
                        details.append(signal)
                    
                    # Adjust score based on signal strength
                    bullish = combined.get('bullish', 0)
                    bearish = combined.get('bearish', 0)
                    
                    if is_call:
                        signal_strength = bullish / (bullish + bearish + 0.001)
                    else:
                        signal_strength = bearish / (bullish + bearish + 0.001)
                    
                    # Add signal strength to score
                    score += signal_strength * 0.3
                    details.append(f"Signal strength for {option_type}: +{signal_strength * 0.3:.2f}")
                    
                else:
                    # No multi-timeframe analysis available
                    score += 0.3  # Default moderate score
                    details.append("No technical analysis data available, using default score: +0.30")
            
            except Exception as e:
                self.logger.error(f"Error in technical analysis scoring: {str(e)}")
                score += 0.3  # Default moderate score
                details.append(f"Error in technical analysis: {str(e)}")
        else:
            # No multi-timeframe analyzer available
            score += 0.3  # Default moderate score
            details.append("No technical analyzer available, using default score: +0.30")
        
        # Normalize score to 0-1 range
        score = min(max(score, 0), 1)
        
        return score, details
    
    def _score_profit_potential(self, symbol, option_data, market_data, risk_tolerance):
        """
        Score profit potential factors
        
        Args:
            symbol (str): The underlying symbol
            option_data (dict): Option data
            market_data (dict): Market data
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            tuple: (score, details)
        """
        score = 0.0
        details = []
        
        # If we have profit predictor, use it
        if self.profit_predictor:
            try:
                # Analyze profit potential
                analysis = self.profit_predictor.analyze_option_profit_potential(option_data, risk_tolerance)
                
                if analysis and 'profit_score' in analysis:
                    # Get profit score (0-100)
                    profit_score = analysis['profit_score'] / 100  # Convert to 0-1 scale
                    score += profit_score * 0.6  # 60% weight to profit score
                    details.append(f"Profit potential score: +{profit_score * 0.6:.2f}")
                    
                    # Add probability of target
                    if 'profit_probability' in analysis and 'probability_of_target' in analysis['profit_probability']:
                        prob = analysis['profit_probability']['probability_of_target']
                        score += prob * 0.2  # 20% weight to probability
                        details.append(f"Probability of reaching target profit: {prob:.1%} (+{prob * 0.2:.2f})")
                    
                    # Add expected return
                    if 'profit_probability' in analysis and 'expected_return_pct' in analysis['profit_probability']:
                        ret = analysis['profit_probability']['expected_return_pct'] / 100  # Convert to 0-1 scale
                        ret = min(max(ret, 0), 1)  # Clamp to 0-1
                        score += ret * 0.1  # 10% weight to expected return
                        details.append(f"Expected return: {ret * 100:.1f}% (+{ret * 0.1:.2f})")
                    
                    # Add win rate
                    if 'profit_probability' in analysis and 'win_rate' in analysis['profit_probability']:
                        win_rate = analysis['profit_probability']['win_rate']
                        score += win_rate * 0.1  # 10% weight to win rate
                        details.append(f"Win rate: {win_rate:.1%} (+{win_rate * 0.1:.2f})")
                    
                else:
                    # No profit analysis available
                    score += 0.3  # Default moderate score
                    details.append("No profit analysis data available, using default score: +0.30")
            
            except Exception as e:
                self.logger.error(f"Error in profit potential scoring: {str(e)}")
                score += 0.3  # Default moderate score
                details.append(f"Error in profit analysis: {str(e)}")
        else:
            # No profit predictor available
            score += 0.3  # Default moderate score
            details.append("No profit predictor available, using default score: +0.30")
        
        # Normalize score to 0-1 range
        score = min(max(score, 0), 1)
        
        return score, details
    
    def _score_market_conditions(self, symbol, option_data, market_data, risk_tolerance):
        """
        Score market condition factors
        
        Args:
            symbol (str): The underlying symbol
            option_data (dict): Option data
            market_data (dict): Market data
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            tuple: (score, details)
        """
        score = 0.0
        details = []
        
        # Get option type (call or put)
        option_type = option_data.get('optionType', '').lower()
        is_call = option_type == 'call'
        
        # Check if market data is available
        if market_data:
            try:
                # Check market trend
                market_trend = market_data.get('market_trend', 'neutral')
                
                # Calculate alignment with option type
                if (is_call and market_trend == 'bullish') or (not is_call and market_trend == 'bearish'):
                    # Market trend aligns with option type
                    score += 0.6
                    details.append(f"Market trend ({market_trend}) aligns with {option_type} option: +0.60")
                elif market_trend == 'neutral':
                    # Neutral market trend
                    score += 0.3
                    details.append(f"Market trend is neutral: +0.30")
                else:
                    # Market trend contradicts option type
                    score += 0.1
                    details.append(f"Market trend ({market_trend}) contradicts {option_type} option: +0.10")
                
                # Check volatility
                vix = market_data.get('vix', 0)
                if vix > 0:
                    # High volatility favors options trading
                    if vix > 30:
                        vol_score = 0.8
                        details.append(f"High market volatility (VIX: {vix:.1f}) favors options trading: +0.80")
                    elif vix > 20:
                        vol_score = 0.6
                        details.append(f"Moderate market volatility (VIX: {vix:.1f}) is favorable: +0.60")
                    elif vix > 15:
                        vol_score = 0.4
                        details.append(f"Normal market volatility (VIX: {vix:.1f}): +0.40")
                    else:
                        vol_score = 0.2
                        details.append(f"Low market volatility (VIX: {vix:.1f}) may limit options profits: +0.20")
                    
                    # Add volatility score
                    score = (score + vol_score) / 2  # Average with previous score
                
                # Check sector performance
                sector_perf = market_data.get('sector_performance', 0)
                if sector_perf != 0:
                    if (is_call and sector_perf > 0) or (not is_call and sector_perf < 0):
                        # Sector performance aligns with option type
                        sector_score = min(abs(sector_perf) / 2, 0.8)  # Cap at 0.8
                        details.append(f"Sector performance ({sector_perf:.1f}%) aligns with {option_type} option: +{sector_score:.2f}")
                    else:
                        # Sector performance contradicts option type
                        sector_score = 0.2
                        details.append(f"Sector performance ({sector_perf:.1f}%) contradicts {option_type} option: +{sector_score:.2f}")
                    
                    # Add sector score
                    score = (score + sector_score) / 2  # Average with previous score
                
            except Exception as e:
                self.logger.error(f"Error in market conditions scoring: {str(e)}")
                score += 0.3  # Default moderate score
                details.append(f"Error in market conditions analysis: {str(e)}")
        else:
            # No market data available
            score += 0.3  # Default moderate score
            details.append("No market data available, using default score: +0.30")
        
        # Normalize score to 0-1 range
        score = min(max(score, 0), 1)
        
        return score, details
    
    def _score_option_metrics(self, symbol, option_data, market_data, risk_tolerance):
        """
        Score option-specific metrics
        
        Args:
            symbol (str): The underlying symbol
            option_data (dict): Option data
            market_data (dict): Market data
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            tuple: (score, details)
        """
        score = 0.0
        details = []
        
        try:
            # Check implied volatility
            iv = option_data.get('impliedVolatility', 0)
            if iv > 0:
                # Score based on implied volatility level
                if iv > 1.0:  # Very high IV (>100%)
                    iv_score = 0.7
                    details.append(f"Very high implied volatility ({iv:.1%}) indicates significant price movement potential: +0.70")
                elif iv > 0.5:  # High IV (>50%)
                    iv_score = 0.8
                    details.append(f"High implied volatility ({iv:.1%}) is favorable for options trading: +0.80")
                elif iv > 0.3:  # Moderate IV (>30%)
                    iv_score = 0.6
                    details.append(f"Moderate implied volatility ({iv:.1%}): +0.60")
                elif iv > 0.15:  # Low IV (>15%)
                    iv_score = 0.4
                    details.append(f"Low implied volatility ({iv:.1%}) may limit profit potential: +0.40")
                else:  # Very low IV
                    iv_score = 0.2
                    details.append(f"Very low implied volatility ({iv:.1%}) may indicate limited price movement: +0.20")
                
                score += iv_score * 0.4  # 40% weight to IV
            
            # Check option Greeks
            if 'delta' in option_data:
                delta = abs(option_data['delta'])
                
                # Score based on delta (probability of being in-the-money at expiration)
                if delta > 0.7:
                    delta_score = 0.8
                    details.append(f"High delta ({delta:.2f}) indicates high probability of being in-the-money: +0.80")
                elif delta > 0.5:
                    delta_score = 0.7
                    details.append(f"Moderate-high delta ({delta:.2f}) indicates good probability of being in-the-money: +0.70")
                elif delta > 0.3:
                    delta_score = 0.5
                    details.append(f"Moderate delta ({delta:.2f}): +0.50")
                elif delta > 0.1:
                    delta_score = 0.3
                    details.append(f"Low delta ({delta:.2f}) indicates lower probability of being in-the-money: +0.30")
                else:
                    delta_score = 0.1
                    details.append(f"Very low delta ({delta:.2f}) indicates low probability of being in-the-money: +0.10")
                
                score += delta_score * 0.3  # 30% weight to delta
            
            # Check theta (time decay)
            if 'theta' in option_data:
                theta = option_data['theta']
                
                # Normalize theta relative to option price
                if 'entryPrice' in option_data and option_data['entryPrice'] > 0:
                    norm_theta = abs(theta) / option_data['entryPrice']
                    
                    # Score based on normalized theta
                    if norm_theta < 0.01:  # Very low daily decay (<1%)
                        theta_score = 0.8
                        details.append(f"Very low time decay ({norm_theta:.2%} per day) is favorable: +0.80")
                    elif norm_theta < 0.02:  # Low daily decay (<2%)
                        theta_score = 0.6
                        details.append(f"Low time decay ({norm_theta:.2%} per day): +0.60")
                    elif norm_theta < 0.04:  # Moderate daily decay (<4%)
                        theta_score = 0.4
                        details.append(f"Moderate time decay ({norm_theta:.2%} per day): +0.40")
                    elif norm_theta < 0.08:  # High daily decay (<8%)
                        theta_score = 0.2
                        details.append(f"High time decay ({norm_theta:.2%} per day) may erode profits quickly: +0.20")
                    else:  # Very high daily decay
                        theta_score = 0.1
                        details.append(f"Very high time decay ({norm_theta:.2%} per day) will erode profits rapidly: +0.10")
                    
                    score += theta_score * 0.2  # 20% weight to theta
            
            # Check bid-ask spread
            if 'bid' in option_data and 'ask' in option_data and option_data['ask'] > 0:
                spread = (option_data['ask'] - option_data['bid']) / option_data['ask']
                
                # Score based on spread percentage
                if spread < 0.03:  # Very tight spread (<3%)
                    spread_score = 0.9
                    details.append(f"Very tight bid-ask spread ({spread:.1%}) indicates high liquidity: +0.90")
                elif spread < 0.05:  # Tight spread (<5%)
                    spread_score = 0.7
                    details.append(f"Tight bid-ask spread ({spread:.1%}) indicates good liquidity: +0.70")
                elif spread < 0.1:  # Moderate spread (<10%)
                    spread_score = 0.5
                    details.append(f"Moderate bid-ask spread ({spread:.1%}): +0.50")
                elif spread < 0.2:  # Wide spread (<20%)
                    spread_score = 0.3
                    details.append(f"Wide bid-ask spread ({spread:.1%}) may impact entry/exit prices: +0.30")
                else:  # Very wide spread
                    spread_score = 0.1
                    details.append(f"Very wide bid-ask spread ({spread:.1%}) indicates poor liquidity: +0.10")
                
                score += spread_score * 0.1  # 10% weight to spread
            
            # If no metrics were available, use default score
            if score == 0:
                score = 0.3  # Default moderate score
                details.append("No option metrics available, using default score: +0.30")
            
        except Exception as e:
            self.logger.error(f"Error in option metrics scoring: {str(e)}")
            score = 0.3  # Default moderate score
            details.append(f"Error in option metrics analysis: {str(e)}")
        
        # Normalize score to 0-1 range
        score = min(max(score, 0), 1)
        
        return score, details
    
    def _generate_recommendation(self, confidence_level, option_type, risk_tolerance):
        """
        Generate recommendation based on confidence level and option type
        
        Args:
            confidence_level (str): Confidence level
            option_type (str): Option type
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            str: Recommendation
        """
        option_type = option_type.lower()
        
        # Define recommendation templates
        templates = {
            'very_high': f"Strong {option_type.upper()} recommendation. High confidence signal with favorable technical and profit indicators.",
            'high': f"{option_type.capitalize()} recommendation. Good confidence with positive indicators.",
            'moderate': f"Consider {option_type}. Moderate confidence with mixed indicators.",
            'low': f"Weak {option_type} signal. Low confidence with some contradicting indicators.",
            'very_low': f"Not recommended. Very low confidence with mostly negative indicators."
        }
        
        # Adjust based on risk tolerance
        if risk_tolerance == 'conservative' and confidence_level not in ['very_high', 'high']:
            return f"Not recommended for conservative investors. Confidence too low."
        elif risk_tolerance == 'aggressive' and confidence_level in ['moderate', 'low']:
            return f"Speculative {option_type} opportunity for aggressive investors only."
        
        return templates.get(confidence_level, f"Neutral on {option_type}.")
    
    def batch_calculate_confidence(self, options_list, market_data=None, risk_tolerance='moderate'):
        """
        Calculate confidence for a batch of options
        
        Args:
            options_list (list): List of option data dictionaries
            market_data (dict): Market data
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            list: List of confidence calculation results
        """
        results = []
        
        for option_data in options_list:
            symbol = option_data.get('symbol', '')
            if not symbol and 'underlying' in option_data:
                symbol = option_data['underlying']
            
            confidence = self.calculate_confidence(symbol, option_data, market_data, risk_tolerance)
            results.append(confidence)
        
        return results
    
    def rank_recommendations(self, options_list, market_data=None, risk_tolerance='moderate', min_confidence=0.3):
        """
        Rank options recommendations by confidence score
        
        Args:
            options_list (list): List of option data dictionaries
            market_data (dict): Market data
            risk_tolerance (str): Risk tolerance level
            min_confidence (float): Minimum confidence score (0-1) to include
            
        Returns:
            list: Ranked list of recommendations
        """
        # Calculate confidence for all options
        confidence_results = self.batch_calculate_confidence(options_list, market_data, risk_tolerance)
        
        # Filter by minimum confidence
        min_score = min_confidence * 100  # Convert to percentage
        filtered_results = [r for r in confidence_results if r['confidence_score'] >= min_score]
        
        # Sort by confidence score (descending)
        ranked_results = sorted(filtered_results, key=lambda x: x['confidence_score'], reverse=True)
        
        return ranked_results
