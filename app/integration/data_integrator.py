"""
Integration module for connecting recommendation components with data sources.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

class DataIntegrator:
    """
    Class to integrate recommendation components with data sources
    """
    
    def __init__(self, data_collector, multi_timeframe_analyzer=None, profit_predictor=None, confidence_calculator=None):
        """
        Initialize the data integrator
        
        Args:
            data_collector: DataCollector instance for retrieving market data
            multi_timeframe_analyzer: MultiTimeframeAnalyzer instance for technical analysis
            profit_predictor: ProfitPredictor instance for profit projections
            confidence_calculator: ConfidenceCalculator instance for confidence calculations
        """
        self.logger = logging.getLogger(__name__)
        self.data_collector = data_collector
        self.multi_timeframe_analyzer = multi_timeframe_analyzer
        self.profit_predictor = profit_predictor
        self.confidence_calculator = confidence_calculator
        
        # Cache for market data to reduce API calls
        self.market_data_cache = {}
        self.cache_expiry = 300  # 5 minutes cache expiry
    
    def get_market_data(self, force_refresh=False):
        """
        Get market data including indices, volatility, and sector performance
        
        Args:
            force_refresh (bool): Whether to force refresh the cache
            
        Returns:
            dict: Market data
        """
        # Check cache first
        now = time.time()
        if not force_refresh and 'market_data' in self.market_data_cache:
            cache_time, data = self.market_data_cache['market_data']
            if now - cache_time < self.cache_expiry:
                return data
        
        try:
            market_data = {}
            
            # Get major indices
            indices = ['SPY', 'QQQ', 'DIA', 'IWM']
            index_data = {}
            
            for index in indices:
                try:
                    data = self.data_collector.get_quote(index)
                    if data is not None:
                        change_pct = data.get('percentChange', 0)
                        index_data[index] = {
                            'price': data.get('lastPrice', 0),
                            'change': data.get('netChange', 0),
                            'change_pct': change_pct
                        }
                except Exception as e:
                    self.logger.error(f"Error getting quote for {index}: {str(e)}")
            
            market_data['indices'] = index_data
            
            # Determine market trend based on SPY
            if 'SPY' in index_data:
                spy_change = index_data['SPY']['change_pct']
                if spy_change > 0.5:
                    market_data['market_trend'] = 'bullish'
                elif spy_change < -0.5:
                    market_data['market_trend'] = 'bearish'
                else:
                    market_data['market_trend'] = 'neutral'
            else:
                market_data['market_trend'] = 'neutral'
            
            # Get VIX (volatility index)
            try:
                vix_data = self.data_collector.get_quote('VIX')
                if vix_data is not None:
                    market_data['vix'] = vix_data.get('lastPrice', 0)
            except Exception as e:
                self.logger.error(f"Error getting VIX data: {str(e)}")
                market_data['vix'] = 0
            
            # Get sector performance (using sector ETFs)
            sectors = {
                'XLK': 'Technology',
                'XLF': 'Financial',
                'XLE': 'Energy',
                'XLV': 'Healthcare',
                'XLI': 'Industrial',
                'XLP': 'Consumer Staples',
                'XLY': 'Consumer Discretionary',
                'XLB': 'Materials',
                'XLU': 'Utilities',
                'XLRE': 'Real Estate'
            }
            
            sector_data = {}
            for symbol, name in sectors.items():
                try:
                    data = self.data_collector.get_quote(symbol)
                    if data is not None:
                        sector_data[name] = {
                            'symbol': symbol,
                            'change_pct': data.get('percentChange', 0)
                        }
                except Exception as e:
                    self.logger.error(f"Error getting quote for {symbol}: {str(e)}")
            
            market_data['sectors'] = sector_data
            
            # Cache the data
            self.market_data_cache['market_data'] = (now, market_data)
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return {}
    
    def get_sector_for_symbol(self, symbol):
        """
        Get the sector for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            str: Sector name
        """
        try:
            # This is a simplified approach - in a real implementation,
            # you would use a more comprehensive database or API
            # to get the sector information
            
            # Technology companies
            tech_companies = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AMZN', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'IBM', 'ADBE', 'CRM', 'PYPL']
            if symbol in tech_companies:
                return 'Technology'
            
            # Financial companies
            financial_companies = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'BLK', 'SCHW']
            if symbol in financial_companies:
                return 'Financial'
            
            # Energy companies
            energy_companies = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'OXY', 'PSX', 'VLO', 'MPC']
            if symbol in energy_companies:
                return 'Energy'
            
            # Healthcare companies
            healthcare_companies = ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'UNH', 'BMY', 'AMGN']
            if symbol in healthcare_companies:
                return 'Healthcare'
            
            # If not found, try to get from data collector
            # (This would require the data collector to have this capability)
            
            # Default to Unknown
            return 'Unknown'
            
        except Exception as e:
            self.logger.error(f"Error getting sector for {symbol}: {str(e)}")
            return 'Unknown'
    
    def get_symbol_market_data(self, symbol):
        """
        Get market data specific to a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Symbol-specific market data
        """
        # Check cache first
        cache_key = f'symbol_data_{symbol}'
        now = time.time()
        if cache_key in self.market_data_cache:
            cache_time, data = self.market_data_cache[cache_key]
            if now - cache_time < self.cache_expiry:
                return data
        
        try:
            symbol_data = {}
            
            # Get quote data
            quote = self.data_collector.get_quote(symbol)
            if quote is not None:
                symbol_data['price'] = quote.get('lastPrice', 0)
                symbol_data['change'] = quote.get('netChange', 0)
                symbol_data['change_pct'] = quote.get('percentChange', 0)
                symbol_data['volume'] = quote.get('totalVolume', 0)
            
            # Get sector
            sector = self.get_sector_for_symbol(symbol)
            symbol_data['sector'] = sector
            
            # Get sector performance
            market_data = self.get_market_data()
            if 'sectors' in market_data and sector in market_data['sectors']:
                symbol_data['sector_performance'] = market_data['sectors'][sector]['change_pct']
            else:
                symbol_data['sector_performance'] = 0
            
            # Get market trend
            symbol_data['market_trend'] = market_data.get('market_trend', 'neutral')
            
            # Get VIX
            symbol_data['vix'] = market_data.get('vix', 0)
            
            # Cache the data
            self.market_data_cache[cache_key] = (now, symbol_data)
            
            return symbol_data
            
        except Exception as e:
            self.logger.error(f"Error getting symbol market data for {symbol}: {str(e)}")
            return {}
    
    def analyze_option(self, symbol, option_data, risk_tolerance='moderate'):
        """
        Perform comprehensive analysis of an option
        
        Args:
            symbol (str): The underlying symbol
            option_data (dict): Option data
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            analysis = {
                'symbol': symbol,
                'option_type': option_data.get('optionType', ''),
                'strike_price': option_data.get('strikePrice', 0),
                'expiration_date': option_data.get('expirationDate', ''),
                'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Get market data
            market_data = self.get_symbol_market_data(symbol)
            analysis['market_data'] = market_data
            
            # Technical analysis
            if self.multi_timeframe_analyzer:
                try:
                    technical_analysis = self.multi_timeframe_analyzer.analyze_multi_timeframe(symbol)
                    analysis['technical_analysis'] = technical_analysis
                except Exception as e:
                    self.logger.error(f"Error in technical analysis for {symbol}: {str(e)}")
                    analysis['technical_analysis'] = {'error': str(e)}
            
            # Profit prediction
            if self.profit_predictor:
                try:
                    profit_analysis = self.profit_predictor.analyze_option_profit_potential(option_data, risk_tolerance)
                    analysis['profit_analysis'] = profit_analysis
                except Exception as e:
                    self.logger.error(f"Error in profit analysis for {symbol}: {str(e)}")
                    analysis['profit_analysis'] = {'error': str(e)}
            
            # Confidence calculation
            if self.confidence_calculator:
                try:
                    confidence = self.confidence_calculator.calculate_confidence(symbol, option_data, market_data, risk_tolerance)
                    analysis['confidence'] = confidence
                except Exception as e:
                    self.logger.error(f"Error in confidence calculation for {symbol}: {str(e)}")
                    analysis['confidence'] = {'error': str(e)}
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing option for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'option_type': option_data.get('optionType', ''),
                'strike_price': option_data.get('strikePrice', 0),
                'error': str(e)
            }
    
    def get_recommendations(self, symbol, expiration_date=None, risk_tolerance='moderate', min_confidence=0.3, max_recommendations=10):
        """
        Get option recommendations for a symbol
        
        Args:
            symbol (str): The stock symbol
            expiration_date (str): Optional expiration date filter
            risk_tolerance (str): Risk tolerance level
            min_confidence (float): Minimum confidence score (0-1)
            max_recommendations (int): Maximum number of recommendations to return
            
        Returns:
            dict: Recommendations including calls and puts
        """
        try:
            # Get option chain
            option_chain = self.data_collector.get_option_chain(symbol, expiration_date)
            if not option_chain or 'callExpDateMap' not in option_chain or 'putExpDateMap' not in option_chain:
                self.logger.error(f"Error getting option chain for {symbol}")
                return {
                    'symbol': symbol,
                    'error': "Failed to retrieve option chain",
                    'calls': [],
                    'puts': []
                }
            
            # Extract options
            all_options = []
            
            # Process calls
            for exp_date, strikes in option_chain['callExpDateMap'].items():
                for strike, options in strikes.items():
                    for option in options:
                        option['optionType'] = 'CALL'
                        all_options.append(option)
            
            # Process puts
            for exp_date, strikes in option_chain['putExpDateMap'].items():
                for strike, options in strikes.items():
                    for option in options:
                        option['optionType'] = 'PUT'
                        all_options.append(option)
            
            # Get market data
            market_data = self.get_symbol_market_data(symbol)
            
            # Calculate confidence for all options
            if self.confidence_calculator:
                ranked_options = self.confidence_calculator.rank_recommendations(
                    all_options, market_data, risk_tolerance, min_confidence
                )
            else:
                # If no confidence calculator, sort by volume as a fallback
                ranked_options = sorted(all_options, key=lambda x: x.get('totalVolume', 0), reverse=True)
            
            # Separate calls and puts
            calls = [opt for opt in ranked_options if opt.get('option_type', '').upper() == 'CALL'][:max_recommendations]
            puts = [opt for opt in ranked_options if opt.get('option_type', '').upper() == 'PUT'][:max_recommendations]
            
            # Enhance recommendations with additional analysis
            enhanced_calls = []
            for call in calls:
                try:
                    option_data = next((opt for opt in all_options if opt.get('optionType') == 'CALL' and opt.get('strikePrice') == call.get('strike_price')), None)
                    if option_data:
                        analysis = self.analyze_option(symbol, option_data, risk_tolerance)
                        enhanced_calls.append({**call, 'analysis': analysis})
                    else:
                        enhanced_calls.append(call)
                except Exception as e:
                    self.logger.error(f"Error enhancing call recommendation: {str(e)}")
                    enhanced_calls.append(call)
            
            enhanced_puts = []
            for put in puts:
                try:
                    option_data = next((opt for opt in all_options if opt.get('optionType') == 'PUT' and opt.get('strikePrice') == put.get('strike_price')), None)
                    if option_data:
                        analysis = self.analyze_option(symbol, option_data, risk_tolerance)
                        enhanced_puts.append({**put, 'analysis': analysis})
                    else:
                        enhanced_puts.append(put)
                except Exception as e:
                    self.logger.error(f"Error enhancing put recommendation: {str(e)}")
                    enhanced_puts.append(put)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'market_data': market_data,
                'risk_tolerance': risk_tolerance,
                'calls': enhanced_calls,
                'puts': enhanced_puts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'calls': [],
                'puts': []
            }
    
    def format_recommendations_for_display(self, recommendations):
        """
        Format recommendations for display in the UI
        
        Args:
            recommendations (dict): Raw recommendations
            
        Returns:
            dict: Formatted recommendations for display
        """
        try:
            formatted = {
                'symbol': recommendations.get('symbol', ''),
                'timestamp': recommendations.get('timestamp', ''),
                'error': recommendations.get('error', None),
                'calls': [],
                'puts': []
            }
            
            # Format calls
            for call in recommendations.get('calls', []):
                formatted_call = self._format_option_for_display(call, 'CALL')
                formatted['calls'].append(formatted_call)
            
            # Format puts
            for put in recommendations.get('puts', []):
                formatted_put = self._format_option_for_display(put, 'PUT')
                formatted['puts'].append(formatted_put)
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting recommendations: {str(e)}")
            return {
                'symbol': recommendations.get('symbol', ''),
                'error': str(e),
                'calls': [],
                'puts': []
            }
    
    def _format_option_for_display(self, option, option_type):
        """
        Format a single option for display
        
        Args:
            option (dict): Option data
            option_type (str): Option type (CALL or PUT)
            
        Returns:
            dict: Formatted option for display
        """
        try:
            # Basic option data
            formatted = {
                'symbol': option.get('symbol', ''),
                'optionType': option_type,
                'strikePrice': option.get('strike_price', option.get('strikePrice', 0)),
                'expirationDate': option.get('expiration_date', option.get('expirationDate', '')),
                'entryPrice': option.get('entry_price', option.get('mark', 0)),
                'bid': option.get('bid', 0),
                'ask': option.get('ask', 0),
                'volume': option.get('totalVolume', 0),
                'openInterest': option.get('openInterest', 0),
                'delta': option.get('delta', 0),
                'gamma': option.get('gamma', 0),
                'theta': option.get('theta', 0),
                'vega': option.get('vega', 0),
                'rho': option.get('rho', 0),
                'impliedVolatility': option.get('volatility', 0) / 100 if option.get('volatility', 0) > 1 else option.get('volatility', 0),
                'daysToExpiration': option.get('daysToExpiration', 0)
            }
            
            # Confidence data
            if 'confidence' in option:
                confidence = option['confidence']
                formatted['confidenceScore'] = confidence.get('confidence_score', 0)
                formatted['confidenceLevel'] = confidence.get('confidence_level', 'unknown')
                formatted['recommendation'] = confidence.get('recommendation', '')
                formatted['signalDetails'] = confidence.get('signal_details', [])
            elif 'analysis' in option and 'confidence' in option['analysis']:
                confidence = option['analysis']['confidence']
                formatted['confidenceScore'] = confidence.get('confidence_score', 0)
                formatted['confidenceLevel'] = confidence.get('confidence_level', 'unknown')
                formatted['recommendation'] = confidence.get('recommendation', '')
                formatted['signalDetails'] = confidence.get('signal_details', [])
            else:
                formatted['confidenceScore'] = 0
                formatted['confidenceLevel'] = 'unknown'
                formatted['recommendation'] = ''
                formatted['signalDetails'] = []
            
            # Profit analysis
            if 'profit_analysis' in option:
                profit = option['profit_analysis']
                formatted['profitScore'] = profit.get('profit_score', 0)
                formatted['expectedReturn'] = profit.get('profit_probability', {}).get('expected_return_pct', 0) if 'profit_probability' in profit else 0
                formatted['winRate'] = profit.get('profit_probability', {}).get('win_rate', 0) if 'profit_probability' in profit else 0
                formatted['takeProfitPrice'] = profit.get('exit_strategy', {}).get('take_profit_price', 0) if 'exit_strategy' in profit else 0
                formatted['stopLossPrice'] = profit.get('exit_strategy', {}).get('stop_loss_price', 0) if 'exit_strategy' in profit else 0
                formatted['optimalHoldDays'] = profit.get('exit_strategy', {}).get('optimal_hold_days', 0) if 'exit_strategy' in profit else 0
            elif 'analysis' in option and 'profit_analysis' in option['analysis']:
                profit = option['analysis']['profit_analysis']
                formatted['profitScore'] = profit.get('profit_score', 0)
                formatted['expectedReturn'] = profit.get('profit_probability', {}).get('expected_return_pct', 0) if 'profit_probability' in profit else 0
                formatted['winRate'] = profit.get('profit_probability', {}).get('win_rate', 0) if 'profit_probability' in profit else 0
                formatted['takeProfitPrice'] = profit.get('exit_strategy', {}).get('take_profit_price', 0) if 'exit_strategy' in profit else 0
                formatted['stopLossPrice'] = profit.get('exit_strategy', {}).get('stop_loss_price', 0) if 'exit_strategy' in profit else 0
                formatted['optimalHoldDays'] = profit.get('exit_strategy', {}).get('optimal_hold_days', 0) if 'exit_strategy' in profit else 0
            else:
                formatted['profitScore'] = 0
                formatted['expectedReturn'] = 0
                formatted['winRate'] = 0
                formatted['takeProfitPrice'] = 0
                formatted['stopLossPrice'] = 0
                formatted['optimalHoldDays'] = 0
            
            # Technical analysis
            if 'technical_analysis' in option and 'combined_signals' in option['technical_analysis']:
                technical = option['technical_analysis']['combined_signals']
                formatted['technicalSentiment'] = technical.get('overall_sentiment', 'neutral')
                formatted['technicalConfidence'] = technical.get('confidence', 0)
            elif 'analysis' in option and 'technical_analysis' in option['analysis'] and 'combined_signals' in option['analysis']['technical_analysis']:
                technical = option['analysis']['technical_analysis']['combined_signals']
                formatted['technicalSentiment'] = technical.get('overall_sentiment', 'neutral')
                formatted['technicalConfidence'] = technical.get('confidence', 0)
            else:
                formatted['technicalSentiment'] = 'neutral'
                formatted['technicalConfidence'] = 0
            
            # Risk/reward ratio
            if formatted['takeProfitPrice'] > 0 and formatted['stopLossPrice'] > 0 and formatted['entryPrice'] > 0:
                potential_profit = formatted['takeProfitPrice'] - formatted['entryPrice']
                potential_loss = formatted['entryPrice'] - formatted['stopLossPrice']
                if potential_loss > 0:
                    formatted['riskRewardRatio'] = potential_profit / potential_loss
                else:
                    formatted['riskRewardRatio'] = 0
            else:
                formatted['riskRewardRatio'] = 0
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting option for display: {str(e)}")
            return {
                'symbol': option.get('symbol', ''),
                'optionType': option_type,
                'strikePrice': option.get('strikePrice', 0),
                'error': str(e)
            }
