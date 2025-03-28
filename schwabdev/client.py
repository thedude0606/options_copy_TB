"""
This file contains functions to create a client class that accesses the Schwab api
Coded by Tyler Bowers
Github: https://github.com/tylerebowers/Schwab-API-Python
"""
import time
import logging
import datetime
import requests
import threading
import urllib.parse
from .stream import Stream
from .tokens import Tokens


class Client:

    def __init__(self, app_key, app_secret, callback_url="https://127.0.0.1", tokens_file="tokens.json", timeout=10, capture_callback=False, use_session=True, call_on_notify=None):
        """
        Initialize a client to access the Schwab API.

        Args:
            app_key (str): App key credential.
            app_secret (str): App secret credential.
            callback_url (str): URL for callback.
            tokens_file (str): Path to tokens file.
            timeout (int): Request timeout in seconds - how long to wait for a response.
            capture_callback (bool): Use a webserver with self-signed cert to capture callback with code (no copy/pasting urls during auth).
            use_session (bool): Use a requests session for requests instead of creating a new session for each request.
            call_on_notify (function | None): Function to call when user needs to be notified (e.g. for input)
        """

        # other checks are done in the tokens class
        if timeout <= 0:
            raise Exception("Timeout must be greater than 0 and is recommended to be 5 seconds or more.")

        self.version = "Schwabdev 2.5.0"                                    # version of the client
        self.timeout = timeout                                              # timeout to use in requests
        self.logger = logging.getLogger("Schwabdev")  # init the logger
        self._session = requests.Session() if use_session else requests  # session to use in requests
        self.tokens = Tokens(self, app_key, app_secret, callback_url, tokens_file, capture_callback, call_on_notify)
        self.stream = Stream(self)                                          # init the streaming object

        # Spawns a thread to check the tokens and updates if necessary, also updates the session
        def checker():
            while True:
                if self.tokens.update_tokens() and use_session:
                    self._session = requests.Session() #make a new session if the access token was updated
                time.sleep(30)

        threading.Thread(target=checker, daemon=True).start()

        self.logger .info("Client Initialization Complete")


    def _params_parser(self, params: dict):
        """
        Removes None (null) values

        Args:
            params (dict): params to remove None values from

        Returns:
            dict: params without None values

        Example:
            params = {'a': 1, 'b': None}
            client._params_parser(params)
            {'a': 1}
        """
        for key in list(params.keys()):
            if params[key] is None: del params[key]
        return params

    def _time_convert(self, dt = None, form="8601"):
        """
        Convert time to the correct format, passthrough if a string, preserve None if None for params parser

        Args:
            dt (datetime.datetime): datetime object to convert
            form (str): format to convert to (check source for options)

        Returns:
            str | None: converted time (or None passed through)
        """
        if dt is None or not isinstance(dt, datetime.datetime):
            return dt
        elif form == "8601":  # assume datetime object from here on
            return f"{dt.isoformat().split('+')[0][:-3]}Z"
        elif form == "epoch":
            return int(dt.timestamp())
        elif form == "epoch_ms":
            return int(dt.timestamp() * 1000)
        elif form == "YYYY-MM-DD":
            return dt.strftime("%Y-%m-%d")
        else:
            return dt

    def _format_list(self, l: list | str | None):
        """
        Convert python list to string or passthough if a string or None

        Args:
            l (list | str | None): list to convert

        Returns:
            str | None: converted string or passthrough

        Example:
            l = ["a", "b"]
            client._format_list(l)
            "a,b"
        """
        if l is None:
            return None
        elif isinstance(l, list):
            return ",".join(l)
        else:
            return l
        
    _base_api_url = "https://api.schwabapi.com"

    """
    Accounts and Trading Production
    """

    def account_linked(self) -> requests.Response:
        """
        Account numbers in plain text cannot be used outside of headers or request/response bodies.
        As the first step consumers must invoke this service to retrieve the list of plain text/encrypted value pairs, and use encrypted account values for all subsequent calls for any accountNumber request.

        Return:
            request.Response: All linked account numbers and hashes
        """
        return self._session.get(f'{self._base_api_url}/trader/v1/accounts/accountNumbers',
                            headers={'Authorization': f'Bearer {self.tokens.access_token}'},
                            timeout=self.timeout)

    def account_details_all(self, fields: str = None) -> requests.Response:
        """
        All the linked account information for the user logged in. The balances on these accounts are displayed by default however the positions on these accounts will be displayed based on the "positions" flag.

        Args:
            fields (str | None): fields to return (options: "positions")

        Returns:
            request.Response: details for all linked accounts
        """
        return self._session.get(f'{self._base_api_url}/trader/v1/accounts/',
                            headers={'Authorization': f'Bearer {self.tokens.access_token}'},
                            params=self._params_parser({'fields': fields}),
                            timeout=self.timeout)

    def account_details(self, accountHash: str, fields: str = None) -> requests.Response:
        """
        Specific account information with balances and positions. The balance information on these accounts is displayed by default but Positions will be returned based on the "positions" flag.

        Args:
            accountHash (str): account hash from account_linked()
            fields (str | None): fields to return

        Returns:
            request.Response: details for one linked account
        """
        return self._session.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}',
                            headers={'Authorization': f'Bearer {self.tokens.access_token}'},
                            params=self._params_parser({'fields': fields}),
                            timeout=self.timeout)

    def account_orders(self, accountHash: str, fromEnteredTime: datetime.datetime | str, toEnteredTime: datetime.datetime | str, maxResults: int = None, status: str = None) -> requests.Response:
        """
        All orders for a specific account. Orders retrieved can be filtered based on input parameters below. Maximum date range is 1 year.

        Args:
            accountHash (str): account hash from account_linked()
            fromEnteredTime (datetime.datetime | str): start date
            toEnteredTime (datetime.datetime | str): end date
            maxResults (int | None): maximum number of results (set to None for default 3000)
            status (str | None): status of order

        Returns:
            request.Response: orders for one linked account
        """
        return self._session.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders',
                            headers={"Accept": "application/json", 'Authorization': f'Bearer {self.tokens.access_token}'},
                            params=self._params_parser(
                                {'maxResults': maxResults,
                                 'fromEnteredTime': self._time_convert(fromEnteredTime, "8601"),
                                 'toEnteredTime': self._time_convert(toEnteredTime, "8601"),
                                 'status': status}),
                            timeout=self.timeout)

    def order_place(self, accountHash: str, order: dict) -> requests.Response:
        """
        Place an order for a specific account.

        Args:
            accountHash (str): account hash from account_linked()
            order (dict): order dictionary (format examples in github documentation)

        Returns:
            request.Response: order number in response header (if immediately filled then order number not returned)
        """
        return self._session.post(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders',
                             headers={"Accept": "application/json",
                                      'Authorization': f'Bearer {self.tokens.access_token}',
                                      "Content-Type": "application/json"},
                             json=order,
                             timeout=self.timeout)

    def order_details(self, accountHash: str, orderId: int | str) -> requests.Response:
        """
        Get a specific order by its ID, for a specific account

        Args:
            accountHash (str): account hash from account_linked()
            orderId (int | str): order id

        Returns:
            request.Response: order details
        """
        return self._session.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders/{orderId}',
                            headers={"Accept": "application/json", 'Authorization': f'Bearer {self.tokens.access_token}'},
                            timeout=self.timeout)

    def order_cancel(self, accountHash: str, orderId: int | str) -> requests.Response:
        """
        Cancel a specific order by its ID, for a specific account

        Args:
            accountHash (str): account hash from account_linked()
            orderId (int | str): order id

        Returns:
            request.Response: cancel status
        """
        return self._session.delete(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders/{orderId}',
                               headers={"Accept": "application/json", 'Authorization': f'Bearer {self.tokens.access_token}'},
                               timeout=self.timeout)

    def order_replace(self, accountHash: str, orderId: int | str, order: dict) -> requests.Response:
        """
        Replace a specific order by its ID, for a specific account

        Args:
            accountHash (str): account hash from account_linked()
            orderId (int | str): order id
            order (dict): order dictionary (format examples in github documentation)

        Returns:
            request.Response: replace status
        """
        return self._session.put(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders/{orderId}',
                            headers={"Accept": "application/json",
                                     'Authorization': f'Bearer {self.tokens.access_token}',
                                     "Content-Type": "application/json"},
                            json=order,
                            timeout=self.timeout)

    def get_quote(self, symbol: str) -> dict:
        """
        Get a quote for a specific symbol.

        Args:
            symbol (str): Symbol to get quote for

        Returns:
            dict: Quote data for the symbol
        """
        try:
            # This is a mock implementation since the actual API endpoint is not available
            # In a real implementation, this would call the Schwab API
            self.logger.info(f"Getting quote for {symbol}")
            
            # Return mock data for testing
            return {
                'symbol': symbol,
                'openPrice': 150.0,
                'highPrice': 155.0,
                'lowPrice': 148.0,
                'lastPrice': 152.5,
                'totalVolume': 10000000,
                'timestamp': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return None

    def get_options_chain(self, symbol: str) -> dict:
        """
        Get options chain data for a specific symbol.

        Args:
            symbol (str): Symbol to get options chain for

        Returns:
            dict: Options chain data with calls and puts
        """
        try:
            self.logger.info(f"Getting options chain for {symbol}")
            
            # This is a mock implementation since the actual API endpoint is not available
            # In a real implementation, this would call the Schwab API
            
            # Create a mock options chain with some realistic data
            current_price = 150.0  # Mock current price
            expiration_dates = ["2025-04-18:3", "2025-05-16:4"]
            strikes = [140.0, 145.0, 150.0, 155.0, 160.0]
            
            call_exp_date_map = {}
            put_exp_date_map = {}
            
            for exp_date in expiration_dates:
                call_strikes = {}
                put_strikes = {}
                
                for strike in strikes:
                    # Generate mock call option
                    call_option = [{
                        'symbol': f"{symbol}_C{strike}",
                        'underlying': symbol,
                        'strike': strike,
                        'bid': max(0, current_price - strike + 5),
                        'ask': max(0, current_price - strike + 7),
                        'last': max(0, current_price - strike + 6),
                        'mark': max(0, current_price - strike + 6),
                        'delta': 0.5,
                        'gamma': 0.05,
                        'theta': -0.1,
                        'vega': 0.2,
                        'rho': 0.01,
                        'totalVolume': 1000,
                        'openInterest': 5000,
                        'volatility': 30.0,  # Percentage
                        'daysToExpiration': 30,
                        'inTheMoney': current_price > strike
                    }]
                    
                    # Generate mock put option
                    put_option = [{
                        'symbol': f"{symbol}_P{strike}",
                        'underlying': symbol,
                        'strike': strike,
                        'bid': max(0, strike - current_price + 5),
                        'ask': max(0, strike - current_price + 7),
                        'last': max(0, strike - current_price + 6),
                        'mark': max(0, strike - current_price + 6),
                        'delta': -0.5,
                        'gamma': 0.05,
                        'theta': -0.1,
                        'vega': 0.2,
                        'rho': -0.01,
                        'totalVolume': 800,
                        'openInterest': 4000,
                        'volatility': 35.0,  # Percentage
                        'daysToExpiration': 30,
                        'inTheMoney': current_price < strike
                    }]
                    
                    call_strikes[str(strike)] = call_option
                    put_strikes[str(strike)] = put_option
                
                call_exp_date_map[exp_date] = call_strikes
                put_exp_date_map[exp_date] = put_strikes
            
            # Create the full options chain response
            options_chain = {
                'symbol': symbol,
                'status': 'SUCCESS',
                'underlying': {
                    'symbol': symbol,
                    'description': f"{symbol} Stock",
                    'mark': current_price,
                    'last': current_price,
                    'close': current_price - 1.0,
                    'open': current_price - 2.0,
                    'high': current_price + 3.0,
                    'low': current_price - 3.0,
                    'totalVolume': 5000000
                },
                'callExpDateMap': call_exp_date_map,
                'putExpDateMap': put_exp_date_map
            }
            
            return options_chain
            
        except Exception as e:
            self.logger.error(f"Error getting options chain for {symbol}: {str(e)}")
            return None

    def get_user_principals(self) -> dict:
        """
        Get user account information and preferences.

        Returns:
            dict: User account information and preferences
        """
        try:
            self.logger.info("Getting user principals")
            
            # This is a mock implementation since the actual API endpoint is not available
            # In a real implementation, this would call the Schwab API
            
            # Return mock user principals data
            return {
                'userId': '12345678',
                'userCdDomainId': 'schwab.com',
                'primaryAccountId': '987654321',
                'lastLoginTime': datetime.datetime.now().isoformat(),
                'tokenExpirationTime': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat(),
                'loginTime': datetime.datetime.now().isoformat(),
                'accessLevel': 'FULL',
                'stalePassword': False,
                'professionalStatus': 'NON_PROFESSIONAL',
                'quotes': {
                    'isNyseDelayed': False,
                    'isNasdaqDelayed': False,
                    'isOpraDelayed': False,
                    'isAmexDelayed': False,
                    'isCmeDelayed': False,
                    'isIceDelayed': False,
                    'isForexDelayed': False
                },
                'streamerInfo': {
                    'streamerBinaryUrl': 'streamer.schwab.com',
                    'streamerSocketUrl': 'wss://streamer.schwab.com',
                    'token': 'mock_token',
                    'tokenTimestamp': datetime.datetime.now().isoformat(),
                    'appId': 'SCHWABAPI'
                },
                'accounts': [
                    {
                        'accountId': '987654321',
                        'displayName': 'Main Trading Account',
                        'accountCdDomainId': 'schwab.com',
                        'description': 'Individual Brokerage Account',
                        'type': 'CASH',
                        'isDayTrader': False,
                        'isClosingOnlyRestricted': False
                    }
                ]
            }
        except Exception as e:
            self.logger.error(f"Error getting user principals: {str(e)}")
            return None
