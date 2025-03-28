# API Documentation

## Schwab API Client Implementation

This document provides information about the implemented API methods in the Schwab API client and DataCollector classes.

### Client Class Methods

#### `get_user_principals()`

Gets user account information and preferences.

```python
user_principals = client.get_user_principals()
```

**Returns:**
- `dict`: User account information and preferences including user ID, account details, and streamer information.

**Example Response:**
```python
{
    'userId': '12345678',
    'userCdDomainId': 'schwab.com',
    'primaryAccountId': '987654321',
    'lastLoginTime': '2025-03-28T13:44:40.309986',
    'tokenExpirationTime': '2025-03-28T14:44:40.309986',
    'loginTime': '2025-03-28T13:44:40.309986',
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
        'tokenTimestamp': '2025-03-28T13:44:40.309986',
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
```

#### `get_options_chain(symbol)`

Gets options chain data for a specific symbol.

```python
options_chain = client.get_options_chain("AAPL")
```

**Parameters:**
- `symbol` (str): Symbol to get options chain for (e.g., "AAPL", "SPY")

**Returns:**
- `dict`: Options chain data with calls and puts

**Example Response:**
```python
{
    'symbol': 'AAPL',
    'status': 'SUCCESS',
    'underlying': {
        'symbol': 'AAPL',
        'description': 'AAPL Stock',
        'mark': 150.0,
        'last': 150.0,
        'close': 149.0,
        'open': 148.0,
        'high': 153.0,
        'low': 147.0,
        'totalVolume': 5000000
    },
    'callExpDateMap': {
        '2025-04-18:3': {
            '150.0': [
                {
                    'symbol': 'AAPL_C150.0',
                    'underlying': 'AAPL',
                    'strike': 150.0,
                    'bid': 5.0,
                    'ask': 7.0,
                    'last': 6.0,
                    'mark': 6.0,
                    'delta': 0.5,
                    'gamma': 0.05,
                    'theta': -0.1,
                    'vega': 0.2,
                    'rho': 0.01,
                    'totalVolume': 1000,
                    'openInterest': 5000,
                    'volatility': 30.0,
                    'daysToExpiration': 30,
                    'inTheMoney': true
                }
            ]
        }
    },
    'putExpDateMap': {
        '2025-04-18:3': {
            '150.0': [
                {
                    'symbol': 'AAPL_P150.0',
                    'underlying': 'AAPL',
                    'strike': 150.0,
                    'bid': 5.0,
                    'ask': 7.0,
                    'last': 6.0,
                    'mark': 6.0,
                    'delta': -0.5,
                    'gamma': 0.05,
                    'theta': -0.1,
                    'vega': 0.2,
                    'rho': -0.01,
                    'totalVolume': 800,
                    'openInterest': 4000,
                    'volatility': 35.0,
                    'daysToExpiration': 30,
                    'inTheMoney': false
                }
            ]
        }
    }
}
```

#### `get_quote(symbol)`

Gets a quote for a specific symbol.

```python
quote = client.get_quote("AAPL")
```

**Parameters:**
- `symbol` (str): Symbol to get quote for (e.g., "AAPL", "SPY")

**Returns:**
- `dict`: Quote data for the symbol

**Example Response:**
```python
{
    'symbol': 'AAPL',
    'openPrice': 150.0,
    'highPrice': 155.0,
    'lowPrice': 148.0,
    'lastPrice': 152.5,
    'totalVolume': 10000000,
    'timestamp': '2025-03-28T13:44:40.309986'
}
```

### DataCollector Class Methods

#### `get_price_data(symbol, period_type='day', period=10, frequency_type='minute', frequency=1)`

Gets historical price data for a symbol.

```python
price_data = data_collector.get_price_data("AAPL", period_type='day', period=10, frequency_type='minute', frequency=1)
```

**Parameters:**
- `symbol` (str): The stock symbol to get price data for
- `period_type` (str, optional): The type of period to show (day, month, year, ytd). Defaults to 'day'.
- `period` (int, optional): The number of periods to show. Defaults to 10.
- `frequency_type` (str, optional): The type of frequency with which a new candle is formed (minute, daily, weekly, monthly). Defaults to 'minute'.
- `frequency` (int, optional): The number of the frequency type to use (e.g., 1, 5, 10, 15, 30 for minute). Defaults to 1.

**Returns:**
- `pd.DataFrame`: Historical price data with columns for open, high, low, close, and volume

**Example Response:**
```
                           open      high       low     close    volume
datetime
2025-03-18 09:30:00  148.000000  151.9600  145.0800  150.7494  500000.0
2025-03-18 09:31:00  150.749399  154.7719  147.7344  152.2469  600000.0
2025-03-18 09:32:00  152.246850  156.0520  149.2019  153.7694  700000.0
```

#### `get_technical_indicators(symbol, indicators=None, period=14)`

Calculates technical indicators for a symbol.

```python
indicators = data_collector.get_technical_indicators("AAPL", indicators=['sma', 'rsi'], period=14)
```

**Parameters:**
- `symbol` (str): The stock symbol to calculate indicators for
- `indicators` (list, optional): List of indicators to calculate. Defaults to ['sma', 'ema', 'rsi', 'macd', 'bollinger'].
- `period` (int, optional): Period to use for indicators. Defaults to 14.

**Returns:**
- `pd.DataFrame`: DataFrame with price data and calculated indicators

**Available Indicators:**
- `sma`: Simple Moving Average
- `ema`: Exponential Moving Average
- `rsi`: Relative Strength Index
- `macd`: Moving Average Convergence Divergence
- `bollinger`: Bollinger Bands

## Testing

A test script (`test_api.py`) is provided to test the implemented API methods. This script bypasses token validation to test the functionality.

To run the tests:

```bash
python test_api.py
```

## Implementation Notes

- These implementations use mock data since the actual API endpoints are not available
- The methods follow the pattern of other methods in the client class and include proper error handling and logging
- The DataCollector's `get_price_data` method includes caching to improve performance for frequently accessed data
