"""
StreamingDataManager class for handling real-time data streaming
from Schwab API using WebSocket connection.
"""

import json
import logging
from datetime import datetime
from app.auth import get_client

class StreamingDataManager:
    """
    Class to manage real-time data streaming from Schwab API
    """
    def __init__(self, interactive_auth=False):
        """
        Initialize the streaming data manager
        
        Args:
            interactive_auth (bool): Whether to allow interactive authentication
        """
        self.client = get_client(interactive=interactive_auth)
        self.streamer = self.client.stream
        self.active = False
        self.subscriptions = {}
        self.symbols = []
        self.callbacks = {}
        self.logger = logging.getLogger(__name__)
    
    def start_stream(self, callback=None):
        """
        Start the data stream
        
        Args:
            callback (function): Optional callback function to override default handler
        """
        if self.active:
            self.logger.warning("Stream already active")
            return
        
        if callback:
            self.streamer.start(callback, daemon=False)
        else:
            self.streamer.start(self._stream_handler, daemon=False)
        
        self.active = True
        self.logger.info("Stream started")
    
    def stop_stream(self):
        """
        Stop the data stream
        """
        if not self.active:
            self.logger.warning("Stream not active")
            return
        
        self.streamer.stop(clear_subscriptions=False)
        self.active = False
        self.logger.info("Stream stopped")
    
    def _stream_handler(self, message):
        """
        Default handler for stream messages
        
        Args:
            message (str): JSON message from the stream
        """
        try:
            print(f"DEBUG - StreamingDataManager received message: {message[:200]}...")
            data = json.loads(message)
            
            # Process notifications
            if "notify" in data:
                print(f"DEBUG - Processing notification: {data['notify']}")
                self._process_notification(data["notify"])
            
            # Process data responses
            if "data" in data:
                print(f"DEBUG - Processing data with {len(data['data'])} items")
                self._process_data(data["data"])
            
            # Process command responses
            if "response" in data:
                print(f"DEBUG - Processing response: {data['response']}")
                self._process_response(data["response"])
            
        except Exception as e:
            self.logger.error(f"Error processing stream message: {str(e)}")
            print(f"DEBUG - Error in _stream_handler: {str(e)}")
    
    def _process_notification(self, notification):
        """
        Process notification messages from the stream
        
        Args:
            notification (list): List of notification objects
        """
        for notify in notification:
            heartbeat = notify.get("heartbeat")
            if heartbeat:
                self.logger.debug(f"Heartbeat received: {heartbeat}")
    
    def _process_data(self, data_list):
        """
        Process data messages from the stream
        
        Args:
            data_list (list): List of data objects
        """
        for data in data_list:
            service = data.get("service")
            print(f"DEBUG - StreamingDataManager processing data for service: {service}")
            
            if service and service in self.callbacks:
                content = data.get("content", [])
                print(f"DEBUG - Found callback for {service}, processing {len(content)} content items")
                
                # Debug the content before passing to callback
                if content:
                    for item in content[:2]:  # Show first 2 items to avoid excessive logging
                        symbol = item.get("key", "unknown")
                        fields = item.get("fields", {})
                        print(f"DEBUG - Content item for {symbol}: fields={list(fields.keys())[:10]}...")
                
                self.callbacks[service](content)
            else:
                print(f"DEBUG - No callback registered for service: {service}")
    
    def _process_response(self, response_list):
        """
        Process response messages from the stream
        
        Args:
            response_list (list): List of response objects
        """
        for response in response_list:
            service = response.get("service")
            command = response.get("command")
            self.logger.debug(f"Response received: {service} - {command}")
    
    def subscribe_level1_quotes(self, symbols, callback=None):
        """
        Subscribe to Level 1 quotes for specified symbols
        
        Args:
            symbols (list): List of stock symbols
            callback (function): Callback function for processing quote data
        """
        if not symbols:
            self.logger.warning("No symbols provided for subscription")
            print("DEBUG - No symbols provided for subscription")
            return
        
        print(f"DEBUG - Subscribing to Level 1 quotes for symbols: {symbols}")
        
        # Register callback if provided
        if callback:
            print(f"DEBUG - Registering callback for LEVELONE_EQUITIES service")
            self.callbacks["LEVELONE_EQUITIES"] = callback
        else:
            print(f"DEBUG - No callback provided for LEVELONE_EQUITIES service")
        
        # Add symbols to subscription list
        for symbol in symbols:
            if symbol not in self.symbols:
                self.symbols.append(symbol)
        
        # Create subscription
        fields = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52"
        # Use basic_request instead of add_level1_equity_subscription
        request = self.streamer.basic_request(
            service="LEVELONE_EQUITIES", 
            command="ADD", 
            parameters={
                "keys": ",".join(symbols),
                "fields": fields
            }
        )
        print(f"DEBUG - Sending subscription request: {request}")
        self.streamer.send(request)
        
        self.logger.info(f"Subscribed to Level 1 quotes for: {', '.join(symbols)}")
        print(f"DEBUG - Subscription request sent for symbols: {symbols}")
        print(f"DEBUG - Current callbacks: {list(self.callbacks.keys())}")
    
    def subscribe_option_quotes(self, symbols, callback=None):
        """
        Subscribe to Level 1 option quotes for specified symbols
        
        Args:
            symbols (list): List of option symbols
            callback (function): Callback function for processing option data
        """
        if not symbols:
            self.logger.warning("No option symbols provided for subscription")
            return
        
        # Register callback if provided
        if callback:
            self.callbacks["LEVELONE_OPTIONS"] = callback
        
        # Create subscription
        fields = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42"
        # Use basic_request instead of add_level1_option_subscription
        request = self.streamer.basic_request(
            service="LEVELONE_OPTIONS", 
            command="ADD", 
            parameters={
                "keys": ",".join(symbols),
                "fields": fields
            }
        )
        self.streamer.send(request)
        
        self.logger.info(f"Subscribed to Level 1 option quotes for: {', '.join(symbols)}")
    
    def unsubscribe_all(self):
        """
        Unsubscribe from all data streams
        """
        # Use basic_request instead of remove_level1_equity_subscription
        if self.symbols:
            # Unsubscribe from LEVELONE_EQUITIES
            request = self.streamer.basic_request(
                service="LEVELONE_EQUITIES", 
                command="REMOVE", 
                parameters={
                    "keys": ",".join(self.symbols)
                }
            )
            self.streamer.send(request)
            
            # Unsubscribe from LEVELONE_OPTIONS if we have any option subscriptions
            if "LEVELONE_OPTIONS" in self.callbacks:
                request = self.streamer.basic_request(
                    service="LEVELONE_OPTIONS", 
                    command="REMOVE", 
                    parameters={
                        "keys": ",".join(self.symbols)
                    }
                )
                self.streamer.send(request)
                
        self.symbols = []
        self.callbacks = {}
        self.logger.info("Unsubscribed from all data streams")
    
    def get_stream_status(self):
        """
        Get the current status of the stream
        
        Returns:
            dict: Stream status information
        """
        return {
            "active": self.active,
            "symbols": self.symbols,
            "subscriptions": list(self.callbacks.keys()),
            "timestamp": datetime.now().isoformat()
        }
