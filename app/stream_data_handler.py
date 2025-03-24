"""
Stream data handler for processing real-time data from Schwab API
"""

import json
import logging
from datetime import datetime

class StreamDataHandler:
    """
    Class to process and format streaming data from Schwab API
    """
    def __init__(self):
        """
        Initialize the stream data handler
        """
        self.logger = logging.getLogger(__name__)
        self.data_store = {}
        self.field_maps = self._initialize_field_maps()
    
    def _initialize_field_maps(self):
        """
        Initialize field maps for different data services
        
        Returns:
            dict: Field maps for different services
        """
        # Field maps for LEVELONE_EQUITIES service
        equity_fields = {
            "0": "symbol",
            "1": "bid_price",
            "2": "ask_price",
            "3": "last_price",
            "4": "bid_size",
            "5": "ask_size",
            "6": "ask_id",
            "7": "bid_id",
            "8": "total_volume",
            "9": "last_size",
            "10": "trade_time",
            "11": "quote_time",
            "12": "high_price",
            "13": "low_price",
            "14": "close_price",
            "15": "exchange_id",
            "16": "description",
            "17": "last_id",
            "18": "open_price",
            "19": "net_change",
            "20": "percent_change",
            "21": "exchange_name",
            "22": "security_status",
            "23": "trade_date",
            "24": "mark",
            "25": "quote_date",
            "26": "marginable",
            "27": "shortable",
            "28": "island_bid",
            "29": "island_ask",
            "30": "island_volume",
            "31": "quote_day",
            "32": "trade_day",
            "33": "volatility",
            "34": "description",
            "35": "last_id",
            "36": "digits",
            "37": "open_price",
            "38": "net_change",
            "39": "high_52_week",
            "40": "low_52_week",
            "41": "pe_ratio",
            "42": "dividend_amount",
            "43": "dividend_yield",
            "44": "island_bid_size",
            "45": "island_ask_size",
            "46": "nav",
            "47": "fund_price",
            "48": "exchange_name",
            "49": "dividend_date",
            "50": "regular_market_quote",
            "51": "regular_market_trade",
            "52": "regular_market_last_price"
        }
        
        # Field maps for LEVELONE_OPTIONS service
        option_fields = {
            "0": "symbol",
            "1": "description",
            "2": "bid_price",
            "3": "ask_price",
            "4": "last_price",
            "5": "high_price",
            "6": "low_price",
            "7": "close_price",
            "8": "total_volume",
            "9": "open_interest",
            "10": "volatility",
            "11": "quote_time",
            "12": "trade_time",
            "13": "money_intrinsic_value",
            "14": "quote_day",
            "15": "trade_day",
            "16": "expiration_year",
            "17": "multiplier",
            "18": "digits",
            "19": "open_price",
            "20": "bid_size",
            "21": "ask_size",
            "22": "last_size",
            "23": "net_change",
            "24": "strike_price",
            "25": "contract_type",
            "26": "underlying",
            "27": "expiration_month",
            "28": "deliverables",
            "29": "time_value",
            "30": "expiration_day",
            "31": "days_to_expiration",
            "32": "delta",
            "33": "gamma",
            "34": "theta",
            "35": "vega",
            "36": "rho",
            "37": "security_status",
            "38": "theoretical_option_value",
            "39": "underlying_price",
            "40": "uv_expiration_type",
            "41": "mark",
            "42": "percent_change"
        }
        
        return {
            "LEVELONE_EQUITIES": equity_fields,
            "LEVELONE_OPTIONS": option_fields
        }
    
    def process_message(self, message):
        """
        Process a message from the stream
        
        Args:
            message (str): JSON message from the stream
            
        Returns:
            dict: Processed data
        """
        try:
            data = json.loads(message)
            
            # Process data responses
            if "data" in data:
                return self._process_data(data["data"])
            
            return None
        except Exception as e:
            self.logger.error(f"Error processing stream message: {str(e)}")
            return None
    
    def _process_data(self, data_list):
        """
        Process data messages from the stream
        
        Args:
            data_list (list): List of data objects
            
        Returns:
            dict: Processed data
        """
        processed_data = {}
        
        for data in data_list:
            service = data.get("service")
            if not service or service not in self.field_maps:
                continue
                
            content = data.get("content", [])
            field_map = self.field_maps[service]
            
            for item in content:
                symbol = item.get("key")
                if not symbol:
                    continue
                    
                # Initialize symbol data if not exists
                if symbol not in self.data_store:
                    self.data_store[symbol] = []
                
                # Process fields
                fields = item.get("fields", {})
                processed_fields = {}
                
                for field_id, value in fields.items():
                    if field_id in field_map:
                        field_name = field_map[field_id]
                        processed_fields[field_name] = value
                
                # Add timestamp
                processed_fields["timestamp"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Format data for display
                display_data = self._format_for_display(symbol, processed_fields, service)
                
                # Add to data store (limit to last 100 data points)
                self.data_store[symbol].append(display_data)
                if len(self.data_store[symbol]) > 100:
                    self.data_store[symbol] = self.data_store[symbol][-100:]
                
                # Add to processed data
                processed_data[symbol] = display_data
        
        return processed_data
    
    def _format_for_display(self, symbol, data, service):
        """
        Format data for display
        
        Args:
            symbol (str): Symbol
            data (dict): Raw data
            service (str): Service name
            
        Returns:
            dict: Formatted data
        """
        formatted = {
            "symbol": symbol,
            "timestamp": data.get("timestamp")
        }
        
        # Format based on service type
        if service == "LEVELONE_EQUITIES":
            formatted.update({
                "price": self._format_price(data.get("last_price")),
                "change": self._format_price(data.get("net_change")),
                "percent_change": self._format_percent(data.get("percent_change")),
                "bid": self._format_price(data.get("bid_price")),
                "ask": self._format_price(data.get("ask_price")),
                "volume": self._format_volume(data.get("total_volume")),
                "high": self._format_price(data.get("high_price")),
                "low": self._format_price(data.get("low_price")),
                "open": self._format_price(data.get("open_price")),
                "close": self._format_price(data.get("close_price"))
            })
        elif service == "LEVELONE_OPTIONS":
            formatted.update({
                "price": self._format_price(data.get("last_price")),
                "change": self._format_price(data.get("net_change")),
                "percent_change": self._format_percent(data.get("percent_change")),
                "bid": self._format_price(data.get("bid_price")),
                "ask": self._format_price(data.get("ask_price")),
                "volume": self._format_volume(data.get("total_volume")),
                "open_interest": data.get("open_interest"),
                "delta": data.get("delta"),
                "gamma": data.get("gamma"),
                "theta": data.get("theta"),
                "vega": data.get("vega"),
                "strike": self._format_price(data.get("strike_price")),
                "expiration": self._format_expiration(
                    data.get("expiration_year"),
                    data.get("expiration_month"),
                    data.get("expiration_day")
                )
            })
        
        return formatted
    
    def _format_price(self, price):
        """Format price value"""
        if price is None:
            return 0.0
        try:
            return round(float(price), 2)
        except (ValueError, TypeError):
            return 0.0
    
    def _format_percent(self, percent):
        """Format percent value"""
        if percent is None:
            return 0.0
        try:
            return round(float(percent), 2)
        except (ValueError, TypeError):
            return 0.0
    
    def _format_volume(self, volume):
        """Format volume value"""
        if volume is None:
            return "0"
        try:
            vol = int(volume)
            if vol >= 1000000:
                return f"{vol/1000000:.2f}M"
            elif vol >= 1000:
                return f"{vol/1000:.2f}K"
            return str(vol)
        except (ValueError, TypeError):
            return "0"
    
    def _format_expiration(self, year, month, day):
        """Format expiration date"""
        if None in (year, month, day):
            return None
        try:
            return f"{year}-{month:02d}-{day:02d}"
        except (ValueError, TypeError):
            return f"{year}-{month}-{day}"
    
    def get_data_store(self):
        """
        Get the current data store
        
        Returns:
            dict: Data store
        """
        return self.data_store
    
    def clear_data_store(self):
        """
        Clear the data store
        """
        self.data_store = {}
