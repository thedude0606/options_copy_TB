"""
Enhanced logging and error handling utilities for options recommendation platform.
Provides consistent logging, error tracking, and debugging capabilities.
"""

import os
import sys
import logging
import traceback
import json
import pandas as pd
from datetime import datetime

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('options_platform.log'),
        logging.StreamHandler()
    ]
)

class EnhancedLogger:
    """
    Enhanced logging utility with additional features for debugging and error tracking.
    """
    
    def __init__(self, name, log_level=logging.INFO, log_to_file=True, log_to_console=True):
        """
        Initialize the enhanced logger.
        
        Args:
            name (str): Logger name
            log_level (int): Logging level
            log_to_file (bool): Whether to log to file
            log_to_console (bool): Whether to log to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.name = name
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Add file handler if requested
        if log_to_file:
            file_handler = logging.FileHandler(f'logs/{name}.log')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Initialize error tracking
        self.error_count = 0
        self.warning_count = 0
        self.last_error = None
        self.error_history = []
        
        # Log initialization
        self.logger.info(f"Enhanced logger '{name}' initialized")
    
    def debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message and track it"""
        self.logger.warning(message)
        self.warning_count += 1
    
    def error(self, message, exc_info=None):
        """
        Log an error message and track it
        
        Args:
            message (str): Error message
            exc_info (Exception, optional): Exception object
        """
        if exc_info:
            self.logger.error(message, exc_info=True)
            error_details = {
                'message': message,
                'exception': str(exc_info),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
        else:
            self.logger.error(message)
            error_details = {
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        
        self.error_count += 1
        self.last_error = error_details
        self.error_history.append(error_details)
    
    def critical(self, message, exc_info=None):
        """
        Log a critical error message and track it
        
        Args:
            message (str): Critical error message
            exc_info (Exception, optional): Exception object
        """
        if exc_info:
            self.logger.critical(message, exc_info=True)
            error_details = {
                'message': message,
                'exception': str(exc_info),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat(),
                'level': 'CRITICAL'
            }
        else:
            self.logger.critical(message)
            error_details = {
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'level': 'CRITICAL'
            }
        
        self.error_count += 1
        self.last_error = error_details
        self.error_history.append(error_details)
    
    def log_data_frame(self, df, message="DataFrame contents", max_rows=10):
        """
        Log a pandas DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame to log
            message (str): Message to include with the DataFrame
            max_rows (int): Maximum number of rows to log
        """
        if not isinstance(df, pd.DataFrame):
            self.warning(f"Not a DataFrame: {type(df)}")
            return
            
        if df.empty:
            self.info(f"{message}: Empty DataFrame")
            return
            
        # Log basic info
        self.info(f"{message}: shape={df.shape}, columns={list(df.columns)}")
        
        # Log sample data
        sample = df.head(max_rows)
        self.debug(f"Sample data:\n{sample}")
    
    def log_api_response(self, response, message="API Response"):
        """
        Log an API response
        
        Args:
            response: API response object
            message (str): Message to include with the response
        """
        try:
            # Check if it's a requests Response object
            if hasattr(response, 'status_code'):
                self.info(f"{message}: status_code={response.status_code}")
                
                # Try to parse as JSON
                try:
                    json_data = response.json()
                    self.debug(f"Response JSON: {json.dumps(json_data, indent=2)[:500]}...")
                except Exception:
                    # Not JSON, log text
                    self.debug(f"Response text: {response.text[:500]}...")
            
            # Check if it's a dictionary
            elif isinstance(response, dict):
                self.info(f"{message}: keys={list(response.keys())}")
                self.debug(f"Response dict: {json.dumps(response, indent=2)[:500]}...")
            
            # Other types
            else:
                self.info(f"{message}: type={type(response)}")
                self.debug(f"Response: {str(response)[:500]}...")
                
        except Exception as e:
            self.warning(f"Error logging API response: {str(e)}")
    
    def get_error_summary(self):
        """
        Get a summary of errors and warnings
        
        Returns:
            dict: Error summary
        """
        return {
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'last_error': self.last_error,
            'error_history': self.error_history[-10:] if len(self.error_history) > 10 else self.error_history
        }
    
    def reset_error_tracking(self):
        """Reset error tracking counters and history"""
        self.error_count = 0
        self.warning_count = 0
        self.last_error = None
        self.error_history = []
        self.logger.info("Error tracking reset")


class ErrorHandler:
    """
    Error handling utility with retry logic and fallback mechanisms.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the error handler.
        
        Args:
            logger (EnhancedLogger, optional): Logger to use
        """
        self.logger = logger or EnhancedLogger("error_handler")
    
    def handle_request_error(self, func, *args, max_retries=3, retry_delay=1, **kwargs):
        """
        Execute a function with retry logic for API requests
        
        Args:
            func: Function to execute
            *args: Function arguments
            max_retries (int): Maximum number of retries
            retry_delay (int): Delay between retries in seconds
            **kwargs: Function keyword arguments
            
        Returns:
            The function result or None if all retries fail
        """
        import time
        
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                if retries > 0:
                    self.logger.info(f"Retry attempt {retries}/{max_retries}...")
                    
                result = func(*args, **kwargs)
                
                if retries > 0:
                    self.logger.info(f"Succeeded after {retries} retries")
                    
                return result
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= max_retries:
                    self.logger.warning(f"Request failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay *= 2
                else:
                    self.logger.error(f"Request failed after {max_retries} retries: {str(e)}", exc_info=e)
        
        return None
    
    def safe_execute(self, func, *args, default_return=None, log_exception=True, **kwargs):
        """
        Safely execute a function and handle any exceptions
        
        Args:
            func: Function to execute
            *args: Function arguments
            default_return: Value to return if function fails
            log_exception (bool): Whether to log the exception
            **kwargs: Function keyword arguments
            
        Returns:
            The function result or default_return if it fails
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_exception:
                self.logger.error(f"Error executing {func.__name__}: {str(e)}", exc_info=e)
            return default_return
    
    def with_fallback(self, primary_func, fallback_func, *args, **kwargs):
        """
        Try a primary function and fall back to an alternative if it fails
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function to use if primary fails
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from either primary or fallback function
        """
        try:
            result = primary_func(*args, **kwargs)
            
            # Check if result is valid (not None or empty)
            if result is None or (hasattr(result, '__len__') and len(result) == 0):
                self.logger.warning(f"Primary function {primary_func.__name__} returned empty result, trying fallback")
                return fallback_func(*args, **kwargs)
                
            return result
            
        except Exception as e:
            self.logger.warning(f"Primary function {primary_func.__name__} failed: {str(e)}, trying fallback")
            return fallback_func(*args, **kwargs)
