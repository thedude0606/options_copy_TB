# Real Data Implementation Guide

## Overview
This document provides instructions for setting up and using the options recommendation platform with real data sources. The platform has been updated to use only real data throughout all components, eliminating mock data sources to provide accurate and reliable options trading recommendations with exit strategies.

## Components

### 1. Real Database (OptionsDatabase)
- **File**: `app/database/options_db.py`
- **Purpose**: Provides persistent storage for options and underlying asset data using SQLite
- **Key Features**:
  - Stores options chain data with all greeks and contract details
  - Stores underlying asset price data (open, high, low, close, volume)
  - Provides methods for retrieving historical data for analysis
  - Supports filtering by symbol, date range, and other parameters

### 2. Enhanced Data Collector (OptionsDataCollector)
- **File**: `app/data/options_collector.py`
- **Purpose**: Collects real-time options and underlying asset data from the Schwab API
- **Key Features**:
  - Background thread for continuous data collection
  - Stores collected data in the real database
  - Provides methods for retrieving data for analysis
  - Configurable collection interval and symbols list

### 3. API Configuration (SchwabAPIConfig)
- **File**: `app/config/schwab_api_config.py`
- **Purpose**: Manages Schwab API authentication and configuration
- **Key Features**:
  - Loads API credentials from config file or environment variables
  - Provides methods for testing API connection
  - Handles token management and authentication
  - Supports saving configuration to file

### 4. Dashboard Integration
- **File**: `run_dashboard_with_exit_strategy.py`
- **Purpose**: Main entry point for the dashboard application
- **Key Features**:
  - Initializes all components with real data sources
  - Starts real-time data collection in the background
  - Provides real-time options recommendations with exit strategies
  - Displays recommendations and exit strategies in the dashboard

## Setup Instructions

### 1. API Credentials
To use real data, you must configure your Schwab API credentials using one of these methods:

#### Option A: Environment Variables
Set the following environment variables:
```bash
export app_key="YOUR_SCHWAB_APP_KEY"
export app_secret="YOUR_SCHWAB_APP_SECRET"
export callback_url="YOUR_CALLBACK_URL"  # Optional, defaults to https://127.0.0.1
```

#### Option B: Configuration File
Create a JSON configuration file at `app/config/schwab_api_config.json`:
```json
{
    "app_key": "YOUR_SCHWAB_APP_KEY",
    "app_secret": "YOUR_SCHWAB_APP_SECRET",
    "callback_url": "YOUR_CALLBACK_URL"
}
```

### 2. Database Setup
The SQLite database will be automatically created at `data/options_data.db` when you first run the application. No additional setup is required.

### 3. Symbols Configuration
By default, the system collects data for SPY, QQQ, AAPL, MSFT, and AMZN. To modify this list:

1. Open `run_dashboard_with_exit_strategy.py`
2. Find the OptionsDataCollector initialization
3. Add your desired symbols list:
```python
data_collector = OptionsDataCollector(
    api_client=client, 
    db=db,
    symbols=["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "YOUR_SYMBOL"]
)
```

## Usage

### Running the Dashboard
Start the dashboard with:
```bash
python run_dashboard_with_exit_strategy.py
```

The system will:
1. Initialize the API client with your credentials
2. Test the API connection
3. Start real-time data collection in the background
4. Launch the dashboard with real-time recommendations and exit strategies

### Data Collection
Data collection happens automatically in the background. The system:
- Collects options chain data for configured symbols
- Collects underlying asset price data
- Stores all data in the SQLite database
- Updates data at the configured interval (default: 60 seconds)

### Accessing Real Data
The real data is accessible through:
- The dashboard UI showing recommendations and exit strategies
- Direct database queries (if needed for custom analysis)
- The OptionsDataCollector API for programmatic access

## Troubleshooting

### API Connection Issues
If you encounter API connection issues:
1. Verify your API credentials are correct
2. Check that your Schwab API account is active
3. Ensure you have the necessary permissions for options data
4. Check the log file for specific error messages

### Database Issues
If you encounter database issues:
1. Check that the `data` directory exists and is writable
2. Delete the database file to recreate it if it becomes corrupted
3. Check the log file for specific error messages

### Data Collection Issues
If data collection isn't working:
1. Check the API connection (see above)
2. Verify the symbols you're requesting are valid
3. Check the log file for collection errors
4. Restart the application to reinitialize the collection thread
