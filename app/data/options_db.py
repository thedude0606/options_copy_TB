"""
Options Database Module

This module provides a database for storing historical options and underlying asset data.
It uses SQLite for storage and provides methods for retrieving and storing data.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import logging
import os

logger = logging.getLogger('options_database')

class OptionsDatabase:
    def __init__(self, db_path=None):
        if db_path is None:
            # Create database in the app directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, 'data', 'options_history.db')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
        logger.info(f"Initializing options database at {db_path}")
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Table for options data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS options_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            underlying TEXT,
            strike REAL,
            option_type TEXT,
            expiration_date TEXT,
            bid REAL,
            ask REAL,
            last REAL,
            volume INTEGER,
            open_interest INTEGER,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            rho REAL,
            implied_volatility REAL,
            timestamp TEXT
        )
        ''')
        
        # Table for underlying asset data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS underlying_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            timestamp TEXT
        )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_symbol ON options_data (symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_timestamp ON options_data (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_underlying_symbol ON underlying_data (symbol)')
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def store_options_data(self, options_data):
        """Store options data from Schwab API"""
        if not options_data:
            logger.warning("No options data to store")
            return
            
        df = pd.DataFrame(options_data)
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now().isoformat()
            
        df.to_sql('options_data', self.conn, if_exists='append', index=False)
        logger.info(f"Stored {len(df)} options records")
        
    def store_underlying_data(self, underlying_data):
        """Store underlying asset data from Schwab API"""
        if not underlying_data:
            logger.warning("No underlying data to store")
            return
            
        df = pd.DataFrame(underlying_data)
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now().isoformat()
            
        df.to_sql('underlying_data', self.conn, if_exists='append', index=False)
        logger.info(f"Stored {len(df)} underlying records")
        
    def get_historical_options(self, symbol, start_date=None, end_date=None):
        """Retrieve historical options data for a symbol"""
        query = f"SELECT * FROM options_data WHERE symbol LIKE '{symbol}%'"
        
        if start_date:
            query += f" AND timestamp >= '{start_date}'"
        if end_date:
            query += f" AND timestamp <= '{end_date}'"
            
        query += " ORDER BY timestamp"
        
        try:
            df = pd.read_sql(query, self.conn)
            logger.info(f"Retrieved {len(df)} historical options records for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error retrieving historical options: {e}")
            return pd.DataFrame()
            
    def get_historical_underlying(self, symbol, lookback_days=None, start_date=None, end_date=None):
        """Retrieve historical underlying data for a symbol"""
        query = f"SELECT * FROM underlying_data WHERE symbol = '{symbol}'"
        
        if start_date:
            query += f" AND timestamp >= '{start_date}'"
        if end_date:
            query += f" AND timestamp <= '{end_date}'"
        
        query += " ORDER BY timestamp"
        
        if lookback_days:
            query += f" LIMIT {lookback_days}"
            
        try:
            df = pd.read_sql(query, self.conn)
            logger.info(f"Retrieved {len(df)} historical underlying records for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error retrieving historical underlying data: {e}")
            return pd.DataFrame()
            
    def get_latest_options(self, symbol):
        """Get the latest options data for a symbol"""
        query = f"""
        SELECT * FROM options_data 
        WHERE symbol LIKE '{symbol}%' 
        AND timestamp = (
            SELECT MAX(timestamp) FROM options_data WHERE symbol LIKE '{symbol}%'
        )
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            logger.info(f"Retrieved {len(df)} latest options records for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error retrieving latest options data: {e}")
            return pd.DataFrame()
