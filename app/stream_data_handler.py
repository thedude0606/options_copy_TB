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
    def __init__(self, streaming_manager=None):
        """
        Initialize the stream data handler
        
        Args:
            streaming_manager: StreamingDataManager instance for managing streaming connections
        """
        self.logger = logging.getLogger(__name__)
        self.data_store = {}
        self.field_maps = self._initialize_field_maps()
        self.streaming_manager = streaming_manager
    
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
            print(f"DEBUG - Received stream message: {message[:200]}...")
            data = json.loads(message)
            
            # Process data responses
            if "data" in data:
                print(f"DEBUG - Processing data message with {len(data['data'])} items")
                return self._process_data(data["data"])
            else:
                print(f"DEBUG - Message does not contain 'data' field. Keys: {list(data.keys())}")
            
            return None
        except Exception as e:
            self.logger.error(f"Error processing stream message: {str(e)}")
            print(f"DEBUG - Error processing stream message: {str(e)}")
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
                print(f"DEBUG - Skipping data with unknown service: {service}")
                continue
                
            content = data.get("content", [])
            print(f"DEBUG - Processing {service} data with {len(content)} content items")
            field_map = self.field_maps[service]
            
            for item in content:
                symbol = item.get("key")
                if not symbol:
                    print(f"DEBUG - Skipping item with no symbol key")
                    continue
                
                print(f"DEBUG - Processing data for symbol: {symbol}")
                    
                # Initialize symbol data if not exists
                if symbol not in self.data_store:
                    self.data_store[symbol] = []
                
                # Process fields
                # The fields are directly in the item, not in a nested "fields" property
                print(f"DEBUG - Raw fields for {symbol}: {item}")
                processed_fields = {}
                
                # Process each key-value pair in the item
                for field_id, value in item.items():
                    # Skip the "key" field as it's the symbol
                    if field_id == "key" or field_id == "delayed" or field_id == "assetMainType" or field_id == "assetSubType" or field_id == "cusip":
                        continue
                        
                    # Convert field_id to string if it's not already
                    field_id_str = str(field_id)
                    if field_id_str in field_map:
                        field_name = field_map[field_id_str]
                        processed_fields[field_name] = value
                        print(f"DEBUG - Processed field: {field_id_str} -> {field_name} = {value}")
                
                print(f"DEBUG - Processed fields for {symbol}: {processed_fields}")
                
                # Add timestamp
                processed_fields["timestamp"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Format data for display
                display_data = self._format_for_display(symbol, processed_fields, service)
                print(f"DEBUG - Formatted display data for {symbol}: {display_data}")
                
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
        print(f"DEBUG - Formatting data for display: symbol={symbol}, service={service}")
        print(f"DEBUG - Raw data keys: {list(data.keys())}")
        
        formatted = {
            "symbol": symbol,
            "timestamp": data.get("timestamp")
        }
        
        # Format based on service type
        if service == "LEVELONE_EQUITIES":
            last_price = data.get("last_price")
            net_change = data.get("net_change")
            percent_change = data.get("percent_change")
            
            print(f"DEBUG - EQUITY Raw values: last_price={last_price}, net_change={net_change}, percent_change={percent_change}")
            
            formatted_price = self._format_price(last_price)
            formatted_change = self._format_price(net_change)
            formatted_percent = self._format_percent(percent_change)
            
            print(f"DEBUG - EQUITY Formatted values: price={formatted_price}, change={formatted_change}, percent={formatted_percent}")
            
            formatted.update({
                "price": formatted_price,
                "change": formatted_change,
                "percent_change": formatted_percent,
                "bid": self._format_price(data.get("bid_price")),
                "ask": self._format_price(data.get("ask_price")),
                "volume": self._format_volume(data.get("total_volume")),
                "high": self._format_price(data.get("high_price")),
                "low": self._format_price(data.get("low_price")),
                "open": self._format_price(data.get("open_price")),
                "close": self._format_price(data.get("close_price"))
            })
        elif service == "LEVELONE_OPTIONS":
            last_price = data.get("last_price")
            net_change = data.get("net_change")
            percent_change = data.get("percent_change")
            
            print(f"DEBUG - OPTION Raw values: last_price={last_price}, net_change={net_change}, percent_change={percent_change}")
            
            formatted_price = self._format_price(last_price)
            formatted_change = self._format_price(net_change)
            formatted_percent = self._format_percent(percent_change)
            
            print(f"DEBUG - OPTION Formatted values: price={formatted_price}, change={formatted_change}, percent={formatted_percent}")
            
            formatted.update({
                "price": formatted_price,
                "change": formatted_change,
                "percent_change": formatted_percent,
                "bid": self._format_price(data.get("bid_price")),
                "ask": self._format_price(data.get("ask_price")),
                "volume": self._format_volume(data.get("total_volume")),
                "open_interest": self._format_volume(data.get("open_interest")),
                "delta": self._format_greek(data.get("delta")),
                "gamma": self._format_greek(data.get("gamma")),
                "theta": self._format_greek(data.get("theta")),
                "vega": self._format_greek(data.get("vega")),
                "rho": self._format_greek(data.get("rho")),
                "strike_price": self._format_price(data.get("strike_price")),
                "underlying_price": self._format_price(data.get("underlying_price")),
                "days_to_expiration": data.get("days_to_expiration")
            })
        
        return formatted
    
    def _format_price(self, value):
        """
        Format price value
        
        Args:
            value: Price value
            
        Returns:
            str: Formatted price
        """
        if value is None:
            return "N/A"
        
        try:
            return f"${float(value):.2f}"
        except (ValueError, TypeError):
            return "N/A"
    
    def _format_percent(self, value):
        """
        Format percent value
        
        Args:
            value: Percent value
            
        Returns:
            str: Formatted percent
        """
        if value is None:
            return "N/A"
        
        try:
            return f"{float(value):.2f}%"
        except (ValueError, TypeError):
            return "N/A"
    
    def _format_volume(self, value):
        """
        Format volume value
        
        Args:
            value: Volume value
            
        Returns:
            str: Formatted volume
        """
        if value is None:
            return "N/A"
        
        try:
            volume = int(value)
            if volume >= 1000000:
                return f"{volume/1000000:.2f}M"
            elif volume >= 1000:
                return f"{volume/1000:.2f}K"
            else:
                return str(volume)
        except (ValueError, TypeError):
            return "N/A"
    
    def _format_greek(self, value):
        """
        Format Greek value
        
        Args:
            value: Greek value
            
        Returns:
            str: Formatted Greek
        """
        if value is None:
            return "N/A"
        
        try:
            return f"{float(value):.4f}"
        except (ValueError, TypeError):
            return "N/A"
    
    def subscribe_to_symbol(self, symbol, service="LEVELONE_EQUITIES"):
        """
        Subscribe to a symbol for streaming data
        
        Args:
            symbol (str): Symbol to subscribe to
            service (str): Service to subscribe to
            
        Returns:
            bool: Success status
        """
        if not self.streaming_manager:
            print("DEBUG - Cannot subscribe: No streaming manager available")
            return False
            
        try:
            print(f"DEBUG - Subscribing to {symbol} for {service}")
            self.streaming_manager.subscribe(service, [symbol])
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {str(e)}")
            print(f"DEBUG - Error subscribing to {symbol}: {str(e)}")
            return False
    
    def unsubscribe_from_symbol(self, symbol, service="LEVELONE_EQUITIES"):
        """
        Unsubscribe from a symbol
        
        Args:
            symbol (str): Symbol to unsubscribe from
            service (str): Service to unsubscribe from
            
        Returns:
            bool: Success status
        """
        if not self.streaming_manager:
            print("DEBUG - Cannot unsubscribe: No streaming manager available")
            return False
            
        try:
            print(f"DEBUG - Unsubscribing from {symbol} for {service}")
            self.streaming_manager.unsubscribe(service, [symbol])
            return True
        except Exception as e:
            self.logger.error(f"Error unsubscribing from {symbol}: {str(e)}")
            print(f"DEBUG - Error unsubscribing from {symbol}: {str(e)}")
            return False
    
    def get_latest_data(self, symbol):
        """
        Get the latest data for a symbol
        
        Args:
            symbol (str): Symbol to get data for
            
        Returns:
            dict: Latest data
        """
        if symbol in self.data_store and self.data_store[symbol]:
            return self.data_store[symbol][-1]
        return None
    
    def get_historical_data(self, symbol, limit=100):
        """
        Get historical data for a symbol
        
        Args:
            symbol (str): Symbol to get data for
            limit (int): Maximum number of data points to return
            
        Returns:
            list: Historical data
        """
        if symbol in self.data_store:
            return self.data_store[symbol][-limit:]
        return []
