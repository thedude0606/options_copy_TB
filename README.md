# Schwab Options Dashboard

This project creates a dashboard that pulls real-time and historical options data from the Schwab API. The dashboard allows users to input a symbol and view options data including price, candles, and the Greeks.

## Features

- Real-time options data retrieval
- Historical data visualization
- Options Greeks display
- Interactive symbol search
- Customizable data views

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Schwab API credentials:
```
SCHWAB_APP_KEY=your_app_key
SCHWAB_APP_SECRET=your_app_secret
SCHWAB_CALLBACK_URL=your_callback_url
```
4. Run the dashboard: `python app.py`

## Usage

The dashboard will be available at http://localhost:8050

## Development

This project is based on the [Schwabdev](https://github.com/tylerebowers/Schwabdev) library.

## License

See the LICENSE file for details.
