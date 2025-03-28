"""
Historical Data Manager Module

This module provides a singleton manager for historical options data.
It initializes the database and data collector, and provides access to them.
"""

import logging
import threading
from options_db import OptionsDatabase
from options_collector import OptionsDataCollector

logger = logging.getLogger('historical_data_manager')

class HistoricalDataManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HistoricalDataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, api_client=None, symbols=None, collection_interval=300):
        if self._initialized:
            return
            
        self.api_client = api_client
        self.symbols = symbols or ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"]
        self.collection_interval = collection_interval
        
        # Initialize database
        self.db = OptionsDatabase()
        
        # Initialize collector if API client is provided
        if api_client:
            self.collector = OptionsDataCollector(api_client, self.db, self.symbols, self.collection_interval)
        else:
            self.collector = None
            
        self._initialized = True
        logger.info("Historical data manager initialized")
        
    def start_collection(self):
        """Start the data collection process"""
        if self.collector:
            self.collector.start_collection()
            logger.info("Started historical data collection")
        else:
            logger.warning("Cannot start collection: No API client provided")
            
    def stop_collection(self):
        """Stop the data collection process"""
        if self.collector:
            self.collector.stop_collection()
            logger.info("Stopped historical data collection")
            
    def get_database(self):
        """Get the options database instance"""
        return self.db
