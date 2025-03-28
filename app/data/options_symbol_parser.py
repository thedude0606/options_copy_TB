"""
Options Symbol Parser Module

This module provides utilities for parsing and handling options symbols.
It supports various formats and provides standardized extraction of option details.
"""

import re
import logging
from datetime import datetime

logger = logging.getLogger('options_symbol_parser')

class OptionsSymbolParser:
    """
    Parser for options symbols in various formats.
    Extracts underlying symbol, expiration date, option type, and strike price.
    """
    
    @staticmethod
    def parse_option_symbol(symbol):
        """
        Parse an option symbol into its components.
        
        Supports multiple formats:
        - Standard: SPY_250516P00566000 or SPY   250516P00566000
        - Alternative: SPY250516P566
        
        Args:
            symbol (str): Option symbol to parse
            
        Returns:
            dict: Option details including underlying, expiration_date, option_type, and strike_price
                  Returns None if parsing fails
        """
        if not symbol:
            logger.warning("Empty symbol provided for parsing")
            return None
            
        try:
            # Clean up the symbol (remove extra spaces)
            clean_symbol = symbol.replace(' ', '')
            
            # Try to match standard format: UNDERLYING_YYMMDDP/CSTRIKE
            pattern = r'([A-Z]+)_?(\d{6})([CP])(\d+)'
            match = re.search(pattern, clean_symbol)
            
            if match:
                underlying = match.group(1)
                date_str = match.group(2)
                option_type = 'CALL' if match.group(3) == 'C' else 'PUT'
                strike_str = match.group(4)
                
                # Parse date (YYMMDD)
                year = int('20' + date_str[0:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                expiration_date = datetime(year, month, day).strftime('%Y-%m-%d')
                
                # Parse strike (may need to add decimal point)
                if len(strike_str) > 2:
                    strike_price = float(strike_str) / 1000
                else:
                    strike_price = float(strike_str)
                
                return {
                    'underlying': underlying,
                    'expiration_date': expiration_date,
                    'option_type': option_type,
                    'strike_price': strike_price,
                    'original_symbol': symbol,
                    'standardized_symbol': f"{underlying}_{date_str}{match.group(3)}{strike_str}"
                }
            
            # Try alternative format without underscore
            alt_pattern = r'([A-Z]+)(\d{6})([CP])(\d+)'
            alt_match = re.search(alt_pattern, clean_symbol)
            
            if alt_match:
                underlying = alt_match.group(1)
                date_str = alt_match.group(2)
                option_type = 'CALL' if alt_match.group(3) == 'C' else 'PUT'
                strike_str = alt_match.group(4)
                
                # Parse date (YYMMDD)
                year = int('20' + date_str[0:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                expiration_date = datetime(year, month, day).strftime('%Y-%m-%d')
                
                # Parse strike (may need to add decimal point)
                if len(strike_str) > 2:
                    strike_price = float(strike_str) / 1000
                else:
                    strike_price = float(strike_str)
                
                return {
                    'underlying': underlying,
                    'expiration_date': expiration_date,
                    'option_type': option_type,
                    'strike_price': strike_price,
                    'original_symbol': symbol,
                    'standardized_symbol': f"{underlying}_{date_str}{alt_match.group(3)}{strike_str}"
                }
                
            logger.warning(f"Could not parse option symbol: {symbol}")
            return None
                
        except Exception as e:
            logger.error(f"Error extracting option details from {symbol}: {e}")
            return None
    
    @staticmethod
    def get_underlying_symbol(option_symbol):
        """
        Extract just the underlying symbol from an option symbol.
        
        Args:
            option_symbol (str): Option symbol
            
        Returns:
            str: Underlying symbol or None if parsing fails
        """
        parsed = OptionsSymbolParser.parse_option_symbol(option_symbol)
        if parsed:
            return parsed['underlying']
        return None
    
    @staticmethod
    def is_option_symbol(symbol):
        """
        Check if a symbol is an option symbol.
        
        Args:
            symbol (str): Symbol to check
            
        Returns:
            bool: True if it's an option symbol, False otherwise
        """
        if not symbol:
            return False
            
        # Clean up the symbol
        clean_symbol = symbol.replace(' ', '')
        
        # Check for option patterns
        std_pattern = r'[A-Z]+_?\d{6}[CP]\d+'
        return bool(re.match(std_pattern, clean_symbol))
