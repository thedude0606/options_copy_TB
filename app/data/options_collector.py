"""
Options Data Collector Module

This module provides a data collector for options and underlying asset data.
It uses the Schwab API to collect data and stores it in the options database.
"""

import time
import threading
import logging
import schedule
from datetime import datetime
import pandas as pd

logger = logging.getLogger('options_collector')

class OptionsDataCollector:
    def __init__(self, api_client, db, symbols=None, collection_interval=60):
        self.symbols = symbols or ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"]
        self.collection_interval = collection_interval
        self.client = api_client
        self.db = db
        self.running = False
        logger.info(f"Initialized options collector for symbols: {', '.join(self.symbols)}")
        
    def start_collection(self):
        """Start the data collection process"""
        if self.running:
            logger.warning("Collection already running")
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Started options data collection")
        
    def stop_collection(self):
        """Stop the data collection process"""
        if not self.running:
            logger.warning("Collection not running")
            return
            
        self.running = False
        logger.info("Stopped options data collection")
        
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                self._collect_data()
            except Exception as e:
                logger.error(f"Error collecting data: {e}")
            time.sleep(self.collection_interval)
    
    def _collect_data(self):
        """Collect options and underlying data for all symbols"""
        collection_time = datetime.now().isoformat()
        logger.info(f"Collecting data at {collection_time}")
        
        for symbol in self.symbols:
            try:
                # Get options chain
                options_chain = self.client.get_options_chain(symbol)
                if options_chain:
                    # Process and store options data
                    options_data = self._process_options_chain(options_chain, collection_time)
                    self.db.store_options_data(options_data)
                    logger.info(f"Collected options data for {symbol}")
                    
                # Get underlying data
                # Try regular symbol first
                logger.info(f"Requesting quote for symbol: {symbol}")
                quote = self.client.get_quote(symbol)
                
                # If VIX fails, try with caret prefix (^VIX)
                if not quote and symbol == "VIX":
                    logger.info(f"No quote data received for {symbol}, trying ^{symbol}")
                    quote = self.client.get_quote(f"^{symbol}")
                
                if quote:
                    underlying_data = [{
                        'symbol': symbol,
                        'open': quote.get('openPrice', 0),
                        'high': quote.get('highPrice', 0),
                        'low': quote.get('lowPrice', 0),
                        'close': quote.get('lastPrice', 0),
                        'volume': quote.get('totalVolume', 0),
                        'timestamp': collection_time
                    }]
                    self.db.store_underlying_data(underlying_data)
                    logger.info(f"Collected underlying data for {symbol}")
                else:
                    logger.warning(f"No quote data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                
    def _process_options_chain(self, options_chain, timestamp):
        """Process options chain data into a flat structure"""
        options_data = []
        
        # Process calls
        for expiration, strikes in options_chain.get('callExpDateMap', {}).items():
            for strike, contracts in strikes.items():
                for contract in contracts:
                    option_data = {
                        'symbol': contract.get('symbol', ''),
                        'underlying': contract.get('underlying', ''),
                        'strike': float(strike),
                        'option_type': 'CALL',
                        'expiration_date': expiration.split(':')[0],
                        'bid': contract.get('bid', 0),
                        'ask': contract.get('ask', 0),
                        'last': contract.get('last', 0),
                        'volume': contract.get('totalVolume', 0),
                        'open_interest': contract.get('openInterest', 0),
                        'delta': contract.get('delta', 0),
                        'gamma': contract.get('gamma', 0),
                        'theta': contract.get('theta', 0),
                        'vega': contract.get('vega', 0),
                        'rho': contract.get('rho', 0),
                        'implied_volatility': contract.get('volatility', 0) / 100,  # Convert from percentage
                        'timestamp': timestamp
                    }
                    options_data.append(option_data)
        
        # Process puts
        for expiration, strikes in options_chain.get('putExpDateMap', {}).items():
            for strike, contracts in strikes.items():
                for contract in contracts:
                    option_data = {
                        'symbol': contract.get('symbol', ''),
                        'underlying': contract.get('underlying', ''),
                        'strike': float(strike),
                        'option_type': 'PUT',
                        'expiration_date': expiration.split(':')[0],
                        'bid': contract.get('bid', 0),
                        'ask': contract.get('ask', 0),
                        'last': contract.get('last', 0),
                        'volume': contract.get('totalVolume', 0),
                        'open_interest': contract.get('openInterest', 0),
                        'delta': contract.get('delta', 0),
                        'gamma': contract.get('gamma', 0),
                        'theta': contract.get('theta', 0),
                        'vega': contract.get('vega', 0),
                        'rho': contract.get('rho', 0),
                        'implied_volatility': contract.get('volatility', 0) / 100,  # Convert from percentage
                        'timestamp': timestamp
                    }
                    options_data.append(option_data)
                    
        return options_data
