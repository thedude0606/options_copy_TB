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
        # Enable debug mode
        self.debug = True
    
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
            if self.debug:
                print(f"\n=== RECOMMENDATION ENGINE DEBUG ===")
                print(f"Generating recommendations for {symbol}")
                print(f"Lookback days: {lookback_days}")
                print(f"Confidence threshold: {confidence_threshold}")
            
            # Get historical data
            historical_data = self.data_collector.get_historical_data(
                symbol=symbol,
                period_type='month',
                period=1,
                frequency_type='daily',
                frequency=1
            )
            
            if self.debug:
                print(f"Historical data shape: {historical_data.shape if not historical_data.empty else 'Empty'}")
            
            if historical_data.empty:
                print(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Get options data
            options_data = self.data_collector.get_option_data(symbol)
            
            if self.debug:
                print(f"Options data shape: {options_data.shape if not options_data.empty else 'Empty'}")
                if not options_data.empty:
                    print(f"Options data columns: {options_data.columns.tolist()}")
                    print(f"Sample option data (first row):")
                    print(options_data.iloc[0])
            
            if options_data.empty:
                print(f"No options data available for {symbol}")
                return pd.DataFrame()
            
            # Calculate technical indicators
            if self.debug:
                print(f"Calculating technical indicators...")
            
            indicators = self._calculate_indicators(historical_data)
            
            if self.debug:
                print(f"Indicators calculated: {list(indicators.keys())}")
            
            # Calculate options Greeks and probabilities
            if self.debug:
                print(f"Analyzing options data...")
            
            options_analysis = self._analyze_options(options_data)
            
            if self.debug:
                print(f"Options analysis shape: {options_analysis.shape if not options_analysis.empty else 'Empty'}")
                if not options_analysis.empty:
                    print(f"Options analysis columns: {options_analysis.columns.tolist()}")
            
            # Generate signals based on technical indicators
            if self.debug:
                print(f"Generating signals from indicators...")
            
            signals = self._generate_signals(indicators)
            
            if self.debug:
                print(f"Signal summary: Bullish={signals['bullish']}, Bearish={signals['bearish']}, Neutral={signals['neutral']}")
                print(f"Signal details: {signals['signal_details']}")
            
            # Score options based on signals and options analysis
            if self.debug:
                print(f"Scoring options based on signals and analysis...")
            
            recommendations = self._score_options(options_analysis, signals, confidence_threshold)
            
            if self.debug:
                print(f"Recommendations shape: {recommendations.shape if not recommendations.empty else 'Empty'}")
                if not recommendations.empty:
                    print(f"Recommendations columns: {recommendations.columns.tolist()}")
                    print(f"Top recommendation:")
                    print(recommendations.iloc[0])
                else:
                    print(f"No recommendations generated that meet the confidence threshold")
            
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
        if self.debug:
            print(f"Analyzing options with Greeks calculation...")
        
        # Calculate Greeks
        options_with_greeks = self.options_analysis.calculate_all_greeks(options_data)
        
        if self.debug:
            print(f"Options with Greeks shape: {options_with_greeks.shape if not options_with_greeks.empty else 'Empty'}")
            if not options_with_greeks.empty:
                print(f"Greek columns: {[col for col in options_with_greeks.columns if col in ['delta', 'gamma', 'theta', 'vega']]}")
        
        # Calculate probability of profit
        if self.debug:
            print(f"Calculating probability of profit...")
        
        options_with_prob = self.options_analysis.calculate_probability_of_profit(options_with_greeks)
        
        if self.debug:
            if 'probabilityOfProfit' in options_with_prob.columns:
                print(f"Probability of profit stats: min={options_with_prob['probabilityOfProfit'].min()}, max={options_with_prob['probabilityOfProfit'].max()}, mean={options_with_prob['probabilityOfProfit'].mean()}")
            else:
                print(f"Warning: probabilityOfProfit column not found after calculation")
        
        # Calculate risk-reward ratio
        if self.debug:
            print(f"Calculating risk-reward ratio...")
        
        analyzed_options = self.options_analysis.calculate_risk_reward_ratio(options_with_prob)
        
        if self.debug:
            if 'riskRewardRatio' in analyzed_options.columns:
                print(f"Risk-reward ratio stats: min={analyzed_options['riskRewardRatio'].min()}, max={analyzed_options['riskRewardRatio'].max()}, mean={analyzed_options['riskRewardRatio'].mean()}")
            else:
                print(f"Warning: riskRewardRatio column not found after calculation")
        
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
                # Get the last close price from the historical data
                # Instead of using the index, get the actual close price value
                if isinstance(indicators['bollinger_middle'], pd.Series):
                    close_price = indicators['bollinger_middle'].iloc[-1]
                else:
                    # If it's a DataFrame, get the close price from the appropriate column
                    close_price = indicators['bollinger_middle'].iloc[-1]
                
                upper = indicators['bollinger_upper'].iloc[-1]
                lower = indicators['bollinger_lower'].iloc[-1]
                middle = indicators['bollinger_middle'].iloc[-1]
                
                # Make sure all values are numeric before comparison
                try:
                    close_price_value = float(close_price)
                    upper_value = float(upper)
                    lower_value = float(lower)
                    middle_value = float(middle)
                    
                    if close_price_value < lower_value:
                        signals['bullish'] += 1
                        signals['signal_details']['bollinger'] = f"Bullish (Price: {close_price_value:.2f} < Lower: {lower_value:.2f})"
                    elif close_price_value > upper_value:
                        signals['bearish'] += 1
                        signals['signal_details']['bollinger'] = f"Bearish (Price: {close_price_value:.2f} > Upper: {upper_value:.2f})"
                    else:
                        signals['neutral'] += 1
                        signals['signal_details']['bollinger'] = f"Neutral (Price: {close_price_value:.2f}, Middle: {middle_value:.2f})"
                except (ValueError, TypeError) as e:
                    print(f"Error converting Bollinger Band values to float: {e}")
                    print(f"close_price type: {type(close_price)}, value: {close_price}")
        
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
        if self.debug:
            print(f"Scoring options with confidence threshold: {confidence_threshold}")
            print(f"Options data shape before scoring: {options_data.shape if not options_data.empty else 'Empty'}")
            if not options_data.empty:
                print(f"Options data columns: {options_data.columns.tolist()}")
                print(f"First option before scoring:")
                print(options_data.iloc[0])
        
        if options_data.empty:
            if self.debug:
                print(f"Error: Options data is empty, cannot score")
            return pd.DataFrame()
        
        # Determine overall market direction
        bullish_signals = signals['bullish']
        bearish_signals = signals['bearish']
        neutral_signals = signals['neutral']
        
        total_signals = bullish_signals + bearish_signals + neutral_signals
        if total_signals == 0:
            if self.debug:
                print(f"Error: No signals available (total_signals=0)")
            return pd.DataFrame()
        
        bullish_score = bullish_signals / total_signals
        bearish_score = bearish_signals / total_signals
        neutral_score = neutral_signals / total_signals
        
        if self.debug:
            print(f"Market direction scores: Bullish={bullish_score:.2f}, Bearish={bearish_score:.2f}, Neutral={neutral_score:.2f}")
        
        market_direction = 'neutral'
        if bullish_score > 0.5 and bullish_score > bearish_score:
            market_direction = 'bullish'
        elif bearish_score > 0.5 and bearish_score > bullish_score:
            market_direction = 'bearish'
        
        if self.debug:
            print(f"Determined market direction: {market_direction}")
        
        # Filter options based on market direction
        if market_direction == 'bullish':
            # For bullish market, recommend calls or put credit spreads
            filtered_options = options_data[options_data['optionType'] == 'CALL']
            if self.debug:
                print(f"Filtered for CALL options in bullish market")
        elif market_direction == 'bearish':
            # For bearish market, recommend puts or call credit spreads
            filtered_options = options_data[options_data['optionType'] == 'PUT']
            if self.debug:
                print(f"Filtered for PUT options in bearish market")
        else:
            # For neutral market, recommend iron condors or straddles
            filtered_options = options_data
            if self.debug:
                print(f"Using all options in neutral market")
        
        if filtered_options.empty:
            if self.debug:
                print(f"Error: No options left after filtering by market direction")
            return pd.DataFrame()
        
        if self.debug:
            print(f"Options data shape after filtering: {filtered_options.shape}")
        
        # Score each option
        scores = []
        for idx, row in filtered_options.iterrows():
            if self.debug and idx == 0:
                print(f"Scoring first option (index {idx}):")
                print(f"Option: {row['symbol'] if 'symbol' in row else 'Unknown'}, Type: {row['optionType'] if 'optionType' in row else 'Unknown'}, Strike: {row['strikePrice'] if 'strikePrice' in row else 'Unknown'}")
            
            score = 0
            
            # Base score from market direction
            if market_direction == 'bullish' and row['optionType'] == 'CALL':
                score += bullish_score * 30  # 30% weight
                if self.debug and idx == 0:
                    print(f"  Added bullish score: +{bullish_score * 30:.2f}")
            elif market_direction == 'bearish' and row['optionType'] == 'PUT':
                score += bearish_score * 30  # 30% weight
                if self.debug and idx == 0:
                    print(f"  Added bearish score: +{bearish_score * 30:.2f}")
            elif market_direction == 'neutral':
                score += neutral_score * 30  # 30% weight
                if self.debug and idx == 0:
                    print(f"  Added neutral score: +{neutral_score * 30:.2f}")
            
            # Score based on probability of profit
            if 'probabilityOfProfit' in row and not pd.isna(row['probabilityOfProfit']):
                pop_score = row['probabilityOfProfit']
                score += pop_score * 30  # 30% weight
                if self.debug and idx == 0:
                    print(f"  Added probability score: +{pop_score * 30:.2f} (POP: {pop_score:.2f})")
            else:
                if self.debug and idx == 0:
                    print(f"  Warning: probabilityOfProfit not available")
            
            # Score based on risk-reward ratio
            if 'riskRewardRatio' in row and not pd.isna(row['riskRewardRatio']):
                rr_ratio = row['riskRewardRatio']
                if rr_ratio > 0:
                    rr_score = min(rr_ratio / 3, 1)  # Cap at 1
                    score += rr_score * 20  # 20% weight
                    if self.debug and idx == 0:
                        print(f"  Added risk-reward score: +{rr_score * 20:.2f} (RR: {rr_ratio:.2f})")
            else:
                if self.debug and idx == 0:
                    print(f"  Warning: riskRewardRatio not available")
            
            # Score based on delta (prefer 0.3-0.7 range)
            if 'delta' in row and not pd.isna(row['delta']):
                delta = abs(row['delta'])
                if 0.3 <= delta <= 0.7:
                    delta_score = 1 - abs(delta - 0.5) / 0.5
                else:
                    delta_score = 0.2  # Lower score for very low or high delta
                
                score += delta_score * 20  # 20% weight
                if self.debug and idx == 0:
                    print(f"  Added delta score: +{delta_score * 20:.2f} (Delta: {delta:.2f})")
            else:
                if self.debug and idx == 0:
                    print(f"  Warning: delta not available")
            
            # Score based on days to expiration
            if 'daysToExpiration' in row and not pd.isna(row['daysToExpiration']):
                # Prefer 30-45 DTE
                # Convert Timestamp to days if it's a Timestamp object
                if isinstance(row['daysToExpiration'], pd.Timedelta):
                    days = row['daysToExpiration'].days
                else:
                    # If it's already a number, use it directly
                    try:
                        days = float(row['daysToExpiration'])
                    except (ValueError, TypeError):
                        days = 0
                        if self.debug and idx == 0:
                            print(f"  Warning: Could not convert daysToExpiration to float: {row['daysToExpiration']}")
                
                if 20 <= days <= 60:
                    dte_score = 1 - abs(days - 40) / 40
                else:
                    dte_score = 0.2  # Lower score for very short or long DTE
                
                score += dte_score * 20  # 20% weight
                if self.debug and idx == 0:
                    print(f"  Added DTE score: +{dte_score * 20:.2f} (Days: {days})")
            else:
                if self.debug and idx == 0:
                    print(f"  Warning: daysToExpiration not available")
            
            if self.debug and idx == 0:
                print(f"  Final score: {score:.2f}, Confidence: {score / 100:.2f}")
            
            # Get underlying price
            underlying_price = 0
            if 'underlyingPrice' in row and not pd.isna(row['underlyingPrice']):
                underlying_price = row['underlyingPrice']
            
            # Calculate entry price (mid price)
            entry_price = 0
            if 'bid' in row and 'ask' in row and not pd.isna(row['bid']) and not pd.isna(row['ask']):
                entry_price = (row['bid'] + row['ask']) / 2
            
            # Add to scores list
            scores.append({
                'symbol': row['symbol'] if 'symbol' in row else '',
                'optionType': row['optionType'] if 'optionType' in row else 'UNKNOWN',
                'strikePrice': row['strikePrice'] if 'strikePrice' in row else 0,
                'expirationDate': row['expirationDate'] if 'expirationDate' in row else 'UNKNOWN',
                'bid': row['bid'] if 'bid' in row else 0,
                'ask': row['ask'] if 'ask' in row else 0,
                'entryPrice': entry_price,
                'underlyingPrice': underlying_price,
                'delta': row['delta'] if 'delta' in row else 0,
                'gamma': row['gamma'] if 'gamma' in row else 0,
                'theta': row['theta'] if 'theta' in row else 0,
                'vega': row['vega'] if 'vega' in row else 0,
                'probabilityOfProfit': row['probabilityOfProfit'] if 'probabilityOfProfit' in row else 0,
                'riskRewardRatio': row['riskRewardRatio'] if 'riskRewardRatio' in row else 0,
                'potentialReturn': row['potentialReturn'] if 'potentialReturn' in row else 0,
                'daysToExpiration': days if 'daysToExpiration' in row and not pd.isna(row['daysToExpiration']) else 0,
                'score': score,
                'confidence': score / 100,  # Convert to 0-1 scale
                'marketDirection': market_direction,
                'signalDetails': signals['signal_details']
            })
        
        # Convert to DataFrame and filter by confidence threshold
        recommendations_df = pd.DataFrame(scores)
        
        if self.debug:
            print(f"Created recommendations DataFrame with {len(scores)} rows")
            if not recommendations_df.empty:
                print(f"Recommendations columns: {recommendations_df.columns.tolist()}")
        
        if not recommendations_df.empty:
            # Filter by confidence threshold
            filtered_recommendations = recommendations_df[recommendations_df['confidence'] >= confidence_threshold]
            
            if self.debug:
                print(f"Filtered recommendations by confidence >= {confidence_threshold}: {len(filtered_recommendations)} rows remaining")
            
            # Sort by confidence
            sorted_recommendations = filtered_recommendations.sort_values('confidence', ascending=False)
            
            if self.debug and not sorted_recommendations.empty:
                print(f"Top recommendation after sorting:")
                print(sorted_recommendations.iloc[0])
            
            return sorted_recommendations
        else:
            if self.debug:
                print(f"No recommendations generated")
            
            return pd.DataFrame()
    
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
