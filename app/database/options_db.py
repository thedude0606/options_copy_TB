"""
Real database implementation for options data storage and retrieval.
Uses SQLite for persistent storage of options and underlying asset data.
"""

import os
import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta

class OptionsDatabase:
    """
    Real database implementation for options data using SQLite.
    Provides methods to store and retrieve options and underlying asset data.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the options database.
        
        Args:
            db_path (str, optional): Path to the SQLite database file.
                If not provided, uses 'options_data.db' in the current directory.
        """
        self.logger = logging.getLogger('options_database')
        
        # Set default database path if not provided
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'options_data.db')
        
        self.db_path = db_path
        self.logger.info(f"Initializing options database at {db_path}")
        
        # Create database and tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        conn = None
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create options data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                timestamp TEXT,
                UNIQUE(symbol, timestamp)
            )
            ''')
            
            # Create underlying data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS underlying_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                timestamp TEXT,
                UNIQUE(symbol, timestamp)
            )
            ''')
            
            conn.commit()
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def store_options_data(self, options_data):
        """
        Store options data in the database.
        
        Args:
            options_data (list): List of dictionaries containing options data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not options_data:
            self.logger.warning("No options data to store")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for option in options_data:
                # Prepare data for insertion
                data = (
                    option.get('symbol', ''),
                    option.get('underlying', ''),
                    option.get('strike', 0),
                    option.get('option_type', ''),
                    option.get('expiration_date', ''),
                    option.get('bid', 0),
                    option.get('ask', 0),
                    option.get('last', 0),
                    option.get('volume', 0),
                    option.get('open_interest', 0),
                    option.get('delta', 0),
                    option.get('gamma', 0),
                    option.get('theta', 0),
                    option.get('vega', 0),
                    option.get('rho', 0),
                    option.get('implied_volatility', 0),
                    option.get('timestamp', datetime.now().isoformat())
                )
                
                # Insert or replace data
                cursor.execute('''
                INSERT OR REPLACE INTO options_data 
                (symbol, underlying, strike, option_type, expiration_date, 
                bid, ask, last, volume, open_interest, delta, gamma, theta, 
                vega, rho, implied_volatility, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', data)
            
            conn.commit()
            self.logger.info(f"Stored {len(options_data)} options data records")
            return True
        except Exception as e:
            self.logger.error(f"Error storing options data: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
    
    def store_underlying_data(self, underlying_data):
        """
        Store underlying asset data in the database.
        
        Args:
            underlying_data (list): List of dictionaries containing underlying asset data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not underlying_data:
            self.logger.warning("No underlying data to store")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for data in underlying_data:
                # Prepare data for insertion
                values = (
                    data.get('symbol', ''),
                    data.get('open', 0),
                    data.get('high', 0),
                    data.get('low', 0),
                    data.get('close', 0),
                    data.get('volume', 0),
                    data.get('timestamp', datetime.now().isoformat())
                )
                
                # Insert or replace data
                cursor.execute('''
                INSERT OR REPLACE INTO underlying_data 
                (symbol, open, high, low, close, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', values)
            
            conn.commit()
            self.logger.info(f"Stored {len(underlying_data)} underlying data records")
            return True
        except Exception as e:
            self.logger.error(f"Error storing underlying data: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_options_data(self, symbol=None, start_date=None, end_date=None):
        """
        Retrieve options data from the database.
        
        Args:
            symbol (str, optional): Symbol to filter by
            start_date (str, optional): Start date for filtering (ISO format)
            end_date (str, optional): End date for filtering (ISO format)
            
        Returns:
            list: List of dictionaries containing options data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query based on filters
            query = "SELECT * FROM options_data WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND (symbol = ? OR underlying = ?)"
                params.extend([symbol, symbol])
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            # Execute query and convert to DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            
            # Convert DataFrame to list of dictionaries
            options_data = df.to_dict('records')
            
            self.logger.info(f"Retrieved {len(options_data)} options data records")
            return options_data
        except Exception as e:
            self.logger.error(f"Error retrieving options data: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_underlying_data(self, symbol=None, start_date=None, end_date=None):
        """
        Retrieve underlying asset data from the database.
        
        Args:
            symbol (str, optional): Symbol to filter by
            start_date (str, optional): Start date for filtering (ISO format)
            end_date (str, optional): End date for filtering (ISO format)
            
        Returns:
            list: List of dictionaries containing underlying asset data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query based on filters
            query = "SELECT * FROM underlying_data WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            # Execute query and convert to DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            
            # Convert DataFrame to list of dictionaries
            underlying_data = df.to_dict('records')
            
            self.logger.info(f"Retrieved {len(underlying_data)} underlying data records")
            return underlying_data
        except Exception as e:
            self.logger.error(f"Error retrieving underlying data: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_latest_options_chain(self, underlying_symbol):
        """
        Get the latest options chain for a specific underlying symbol.
        
        Args:
            underlying_symbol (str): Underlying symbol to get options chain for
            
        Returns:
            dict: Dictionary containing calls and puts data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get the latest timestamp for this underlying
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(timestamp) FROM options_data WHERE underlying = ?",
                (underlying_symbol,)
            )
            latest_timestamp = cursor.fetchone()[0]
            
            if not latest_timestamp:
                self.logger.warning(f"No options data found for {underlying_symbol}")
                return {"calls": [], "puts": []}
            
            # Get all options for this underlying at the latest timestamp
            query = """
            SELECT * FROM options_data 
            WHERE underlying = ? AND timestamp = ?
            """
            df = pd.read_sql_query(query, conn, params=(underlying_symbol, latest_timestamp))
            
            # Split into calls and puts
            calls = df[df['option_type'] == 'CALL'].to_dict('records')
            puts = df[df['option_type'] == 'PUT'].to_dict('records')
            
            self.logger.info(f"Retrieved options chain for {underlying_symbol}: {len(calls)} calls, {len(puts)} puts")
            return {"calls": calls, "puts": puts}
        except Exception as e:
            self.logger.error(f"Error retrieving options chain: {str(e)}")
            return {"calls": [], "puts": []}
        finally:
            if conn:
                conn.close()
    
    def get_historical_prices(self, symbol, days=30):
        """
        Get historical price data for a symbol.
        
        Args:
            symbol (str): Symbol to get historical data for
            days (int, optional): Number of days of historical data to retrieve
            
        Returns:
            pandas.DataFrame: DataFrame containing historical price data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Query for historical data
            query = """
            SELECT * FROM underlying_data 
            WHERE symbol = ? AND timestamp >= ? 
            ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(symbol, start_date.isoformat())
            )
            
            if df.empty:
                self.logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.logger.info(f"Retrieved {len(df)} historical price records for {symbol}")
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving historical prices: {str(e)}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
