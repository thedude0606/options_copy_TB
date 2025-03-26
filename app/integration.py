"""
Integration module for connecting all components of the options recommendation platform.
Provides functions for testing and validating the implementation.
"""
import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

# Import custom modules
from app.simplified_layout import create_simplified_layout
from app.components.recommendation_card import create_recommendation_grid, create_recommendation_card
from app.data_pipeline import ShortTermDataPipeline
from app.analysis.short_term_recommendation_engine import ShortTermRecommendationEngine
from app.visualizations.validation_charts import (
    create_validation_chart, 
    create_timeframe_comparison_chart,
    create_sentiment_chart,
    create_risk_reward_visualization
)

# Import tab components and their callback registration functions
from app.components.greeks_tab import create_greeks_tab, register_greeks_callbacks
from app.components.indicators_tab import create_indicators_tab, register_indicators_callbacks
from app.historical_tab import create_historical_tab, register_historical_callbacks
from app.real_time_tab import get_real_time_tab_layout, register_real_time_callbacks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integration')

def initialize_components(app, data_collector):
    """
    Initialize all components of the options recommendation platform
    
    Args:
        app: Dash app instance
        data_collector: DataCollector instance
        
    Returns:
        tuple: (data_pipeline, recommendation_engine)
    """
    logger.info("Initializing components...")
    
    # Initialize data pipeline
    data_pipeline = ShortTermDataPipeline(data_collector)
    logger.info("Data pipeline initialized")
    
    # Initialize recommendation engine
    recommendation_engine = ShortTermRecommendationEngine(data_pipeline)
    logger.info("Recommendation engine initialized")
    
    return data_pipeline, recommendation_engine

def register_callbacks(app, data_pipeline, recommendation_engine):
    """
    Register all callbacks for the simplified dashboard
    
    Args:
        app: Dash app instance
        data_pipeline: ShortTermDataPipeline instance
        recommendation_engine: ShortTermRecommendationEngine instance
    """
    logger.info("Registering callbacks...")
    
    # Register callbacks for each feature tab
    register_greeks_callbacks(app)
    register_indicators_callbacks(app)
    register_historical_callbacks(app)
    register_real_time_callbacks(app)
    
    # Callback for symbol search
    @app.callback(
        [Output("recommendations-container", "children"),
         Output("stored-recommendations", "children")],
        [Input("submit-button", "n_clicks"),
         Input("trading-timeframe", "value"),
         Input("filter-all", "n_clicks"),
         Input("filter-calls", "n_clicks"),
         Input("filter-puts", "n_clicks"),
         Input("apply-settings", "n_clicks")],
        [State("symbol-input", "value"),
         State("recommendation-confidence-threshold", "value"),
         State("recommendation-risk-reward-threshold", "value"),
         State("weight-rsi", "value"),
         State("weight-macd", "value"),
         State("weight-bb", "value"),
         State("stored-recommendations", "children")]
    )
    def update_recommendations(
        search_clicks, timeframe, all_clicks, calls_clicks, puts_clicks, 
        settings_clicks, symbol, confidence, risk_reward, rsi_weight, 
        macd_weight, bb_weight, stored_recommendations
    ):
        """Update recommendations based on user inputs"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return html.Div("Enter a symbol and click Search to view recommendations"), None
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Determine option type filter based on which button was clicked
        option_type = "ALL"  # Default
        if trigger_id == "filter-calls":
            option_type = "CALL"
        elif trigger_id == "filter-puts":
            option_type = "PUT"
        
        # If we're just filtering existing recommendations
        if trigger_id in ["filter-all", "filter-calls", "filter-puts"] and stored_recommendations:
            try:
                all_recommendations = json.loads(stored_recommendations)
                filtered_recommendations = [
                    rec for rec in all_recommendations 
                    if option_type == "ALL" or rec["option_type"] == option_type
                ]
                return create_recommendation_grid(filtered_recommendations), stored_recommendations
            except Exception as e:
                logger.error(f"Error filtering recommendations: {str(e)}")
                return html.Div(f"Error: {str(e)}"), None
        
        # Otherwise, generate new recommendations
        if not symbol:
            return html.Div("Please enter a valid symbol"), None
            
        try:
            # Set up indicator weights
            indicator_weights = {
                "rsi": rsi_weight or 5,
                "macd": macd_weight or 5,
                "bollinger": bb_weight or 5
            }
            
            # Generate recommendations
            recommendations = recommendation_engine.generate_recommendations(
                symbol=symbol,
                timeframe=timeframe or '30m',
                option_type=option_type,
                min_confidence=confidence or 0.6,
                min_risk_reward=risk_reward or 1.5,
                indicator_weights=indicator_weights
            )
            
            if not recommendations:
                return html.Div("No recommendations found for the current criteria"), None
                
            # Filter recommendations if needed
            if option_type != "ALL":
                recommendations = [rec for rec in recommendations if rec["option_type"] == option_type]
                
            # Create recommendation grid
            grid = create_recommendation_grid(recommendations)
            
            # Store all recommendations for filtering
            stored_data = json.dumps(recommendations)
            
            return grid, stored_data
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return html.Div(f"Error: {str(e)}"), None
    
    # Callback for recommendation selection and validation visualization
    @app.callback(
        Output("validation-content", "children"),
        [Input("recommendations-container", "children")],
        [State("trading-timeframe", "value"),
         State("stored-recommendations", "children"),
         State("symbol-input", "value")]
    )
    def update_validation_visualization(recommendation_grid, timeframe, stored_recommendations, symbol):
        """Update validation visualization based on selected recommendation"""
        if not stored_recommendations or not symbol:
            return html.Div("Select a recommendation to view validation details")
            
        try:
            # Get the first recommendation for validation
            recommendations = json.loads(stored_recommendations)
            if not recommendations:
                return html.Div("No recommendations available for validation")
                
            recommendation = recommendations[0]
            
            # Get validation data
            validation_data = recommendation_engine.get_validation_data(recommendation, timeframe or '30m')
            
            # Create validation charts
            validation_chart = create_validation_chart(validation_data)
            risk_reward_chart = create_risk_reward_visualization(recommendation)
            sentiment_chart = create_sentiment_chart(recommendation.get('sentiment', {}))
            
            # Create validation content
            validation_content = html.Div([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=validation_chart, config={'displayModeBar': False})
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=risk_reward_chart, config={'displayModeBar': False})
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=sentiment_chart, config={'displayModeBar': False})
                    ], width=6)
                ])
            ])
            
            return validation_content
            
        except Exception as e:
            logger.error(f"Error updating validation visualization: {str(e)}")
            return html.Div(f"Error: {str(e)}")
    
    # Callback for settings modal
    @app.callback(
        Output("settings-modal", "is_open"),
        [Input("recommendations-settings", "n_clicks"),
         Input("apply-settings", "n_clicks"),
         Input("reset-settings", "n_clicks")],
        [State("settings-modal", "is_open")]
    )
    def toggle_settings_modal(settings_clicks, apply_clicks, reset_clicks, is_open):
        """Toggle settings modal"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open
            
        return not is_open
    
    # Callback for resetting settings
    @app.callback(
        [Output("recommendation-confidence-threshold", "value"),
         Output("recommendation-risk-reward-threshold", "value"),
         Output("weight-rsi", "value"),
         Output("weight-macd", "value"),
         Output("weight-bb", "value")],
        [Input("reset-settings", "n_clicks")]
    )
    def reset_settings(n_clicks):
        """Reset settings to defaults"""
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        return 0.6, 1.5, 5, 5, 5
    
    # Callback for validation collapse
    @app.callback(
        Output("validation-collapse", "is_open"),
        [Input("expand-validation", "n_clicks")],
        [State("validation-collapse", "is_open")]
    )
    def toggle_validation_collapse(n_clicks, is_open):
        """Toggle validation collapse"""
        if n_clicks:
            return not is_open
        return is_open
    
    # Callback for market overview
    @app.callback(
        Output("market-overview-content", "children"),
        [Input("submit-button", "n_clicks")],
        [State("symbol-input", "value")]
    )
    def update_market_overview(n_clicks, symbol):
        """Update market overview"""
        if not n_clicks:
            # Get initial market overview
            try:
                market_data = data_pipeline.get_market_overview()
                
                if not market_data:
                    return html.P("Market data not available", className="text-muted")
                    
                # Create market overview content
                overview_content = []
                
                for index, data in market_data.items():
                    change = data.get('change', 0)
                    change_pct = data.get('change_percent', 0)
                    
                    color = "success" if change >= 0 else "danger"
                    
                    overview_content.append(
                        dbc.Row([
                            dbc.Col(html.Span(index), width=3),
                            dbc.Col(html.Span(f"${data.get('last_price', 0):.2f}"), width=4),
                            dbc.Col(
                                html.Span(
                                    f"{change:+.2f} ({change_pct:+.2f}%)",
                                    className=f"text-{color}"
                                ), 
                                width=5
                            )
                        ], className="mb-2")
                    )
                
                return html.Div([
                    dbc.Row([
                        dbc.Col(html.Strong("Index"), width=3),
                        dbc.Col(html.Strong("Price"), width=4),
                        dbc.Col(html.Strong("Change"), width=5)
                    ], className="mb-2"),
                    html.Hr(className="my-1"),
                    html.Div(overview_content)
                ])
                
            except Exception as e:
                logger.error(f"Error updating market overview: {str(e)}")
                return html.P(f"Error: {str(e)}", className="text-danger")
        
        return dash.no_update
    
    # Callback for feature tabs content
    @app.callback(
        Output("feature-content", "children"),
        [Input("feature-tabs", "active_tab"),
         Input("symbol-input", "value")]
    )
    def update_feature_content(active_tab, symbol):
        """Update feature content based on selected tab"""
        if not symbol:
            return html.Div([
                html.P("Please enter a symbol and click Search to view data.")
            ])
            
        if active_tab == "tab-options-chain":
            try:
                # Get options chain data from data pipeline
                options_data = data_pipeline.get_options_data_for_timeframe(symbol)
                
                if not options_data or not options_data.get('options'):
                    return html.Div([
                        html.P(f"No options data available for {symbol}. Please try another symbol.")
                    ])
                    
                # Create options chain table
                options_list = options_data.get('options', [])
                
                # Create DataFrame for display
                df = pd.DataFrame(options_list)
                
                # Create the options chain table
                return html.Div([
                    html.H5(f"Options Chain for {symbol}"),
                    dash_table.DataTable(
                        id='options-chain-table',
                        columns=[
                            {"name": "Type", "id": "option_type"},
                            {"name": "Strike", "id": "strike", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Expiration", "id": "expiration"},
                            {"name": "Bid", "id": "bid", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Ask", "id": "ask", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Last", "id": "last", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Volume", "id": "volume", "type": "numeric"},
                            {"name": "Open Int", "id": "open_interest", "type": "numeric"}
                        ],
                        data=df.to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center'},
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'option_type', 'filter_query': '{option_type} eq "CALL"'},
                                'backgroundColor': '#e6f3ff',
                                'color': '#0066cc'
                            },
                            {
                                'if': {'column_id': 'option_type', 'filter_query': '{option_type} eq "PUT"'},
                                'backgroundColor': '#ffe6e6',
                                'color': '#cc0000'
                            }
                        ],
                        page_size=10
                    )
                ])
                
            except Exception as e:
                logger.error(f"Error displaying options chain: {str(e)}")
                return html.Div([
                    html.P(f"Error retrieving options chain: {str(e)}")
                ])
                
        elif active_tab == "tab-greeks":
            # Return the Greeks tab content
            try:
                # Pass the current symbol to the Greeks tab
                greeks_content = create_greeks_tab()
                
                # Add a script to automatically fill the symbol input and click analyze
                auto_fill_script = html.Script(f'''
                    setTimeout(function() {{
                        var symbolInput = document.getElementById("greeks-symbol-input");
                        var analyzeButton = document.getElementById("greeks-analyze-button");
                        if (symbolInput && analyzeButton) {{
                            symbolInput.value = "{symbol}";
                            analyzeButton.click();
                        }}
                    }}, 500);
                ''')
                
                return html.Div([
                    greeks_content,
                    auto_fill_script
                ])
            except Exception as e:
                logger.error(f"Error displaying Greeks tab: {str(e)}")
                return html.Div([
                    html.P(f"Error displaying Greeks tab: {str(e)}")
                ])
            
        elif active_tab == "tab-indicators":
            # Return the Technical Indicators tab content
            try:
                # Pass the current symbol to the Indicators tab
                indicators_content = create_indicators_tab()
                
                # Add a script to automatically fill the symbol input and click analyze
                auto_fill_script = html.Script(f'''
                    setTimeout(function() {{
                        var symbolInput = document.getElementById("indicator-symbol-input");
                        var analyzeButton = document.getElementById("indicator-analyze-button");
                        var updateButton = document.getElementById("update-indicators-button");
                        if (symbolInput && analyzeButton) {{
                            symbolInput.value = "{symbol}";
                            analyzeButton.click();
                            setTimeout(function() {{
                                if (updateButton) {{
                                    updateButton.click();
                                }}
                            }}, 1000);
                        }}
                    }}, 500);
                ''')
                
                return html.Div([
                    indicators_content,
                    auto_fill_script
                ])
            except Exception as e:
                logger.error(f"Error displaying Technical Indicators tab: {str(e)}")
                return html.Div([
                    html.P(f"Error displaying Technical Indicators tab: {str(e)}")
                ])
            
        elif active_tab == "tab-historical":
            # Return the Historical Data tab content
            try:
                # Create the historical tab content
                historical_content = create_historical_tab()
                
                # Add a script to automatically load data for the current symbol
                auto_load_script = html.Script(f'''
                    setTimeout(function() {{
                        var loadButton = document.getElementById("load-historical-data");
                        if (loadButton) {{
                            // Set the current symbol in a hidden input
                            var hiddenInput = document.createElement("input");
                            hiddenInput.type = "hidden";
                            hiddenInput.id = "hidden-symbol-input";
                            hiddenInput.value = "{symbol}";
                            document.body.appendChild(hiddenInput);
                            
                            // Click the load button
                            loadButton.click();
                        }}
                    }}, 500);
                ''')
                
                return html.Div([
                    historical_content,
                    auto_load_script
                ])
            except Exception as e:
                logger.error(f"Error displaying Historical Data tab: {str(e)}")
                return html.Div([
                    html.P(f"Error displaying Historical Data tab: {str(e)}")
                ])
            
        elif active_tab == "tab-realtime":
            # Return the Real-Time Data tab content
            try:
                # Create the real-time tab content
                realtime_content = get_real_time_tab_layout()
                
                # Add a script to automatically add the symbol and start the stream
                auto_start_script = html.Script(f'''
                    setTimeout(function() {{
                        var symbolInput = document.getElementById("rt-symbol-input");
                        var addButton = document.getElementById("rt-add-symbol-button");
                        var startButton = document.getElementById("rt-start-stream-button");
                        
                        if (symbolInput && addButton && startButton) {{
                            symbolInput.value = "{symbol}";
                            addButton.click();
                            
                            setTimeout(function() {{
                                startButton.click();
                            }}, 500);
                        }}
                    }}, 500);
                ''')
                
                return html.Div([
                    realtime_content,
                    auto_start_script
                ])
            except Exception as e:
                logger.error(f"Error displaying Real-Time Data tab: {str(e)}")
                return html.Div([
                    html.P(f"Error displaying Real-Time Data tab: {str(e)}")
                ])
        
        return html.Div([
            html.P("Select a tab to view content.")
        ])
    
    # Add callback to update real-time data
    @app.callback(
        Output("rt-stream-data", "data"),
        [Input("rt-update-interval", "n_intervals")],
        [State("rt-symbols-store", "data"),
         State("rt-connection-store", "data"),
         State("rt-data-type", "value")]
    )
    def update_stream_data(n_intervals, symbols, connection_data, data_type):
        """Update real-time data stream"""
        if not n_intervals or not symbols or not connection_data or not connection_data.get("active", False):
            return {}
            
        try:
            # Get real-time data for each symbol
            data_dict = {}
            
            for symbol in symbols:
                # Get real-time data from data pipeline
                if data_type == "quotes":
                    quote_data = data_pipeline.get_real_time_data(symbol)
                    
                    if quote_data:
                        # Format data for chart
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        price = quote_data.get("last_price", 0)
                        
                        # Initialize symbol data if not exists
                        if symbol not in data_dict:
                            data_dict[symbol] = {
                                "timestamps": [],
                                "prices": [],
                                "volumes": [],
                                "last_price": price,
                                "change": quote_data.get("change", 0),
                                "change_percent": quote_data.get("change_percent", 0),
                                "bid": quote_data.get("bid", 0),
                                "ask": quote_data.get("ask", 0),
                                "volume": quote_data.get("volume", 0)
                            }
                        
                        # Add new data point
                        data_dict[symbol]["timestamps"].append(timestamp)
                        data_dict[symbol]["prices"].append(price)
                        data_dict[symbol]["volumes"].append(quote_data.get("volume", 0))
                        
                        # Keep only last 100 data points
                        if len(data_dict[symbol]["timestamps"]) > 100:
                            data_dict[symbol]["timestamps"] = data_dict[symbol]["timestamps"][-100:]
                            data_dict[symbol]["prices"] = data_dict[symbol]["prices"][-100:]
                            data_dict[symbol]["volumes"] = data_dict[symbol]["volumes"][-100:]
                
                elif data_type == "options":
                    options_data = data_pipeline.get_options_data_for_timeframe(symbol)
                    
                    if options_data and options_data.get("options"):
                        data_dict[symbol] = {
                            "options": options_data.get("options", []),
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "underlying_price": options_data.get("underlying_price", 0)
                        }
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error updating stream data: {str(e)}")
            return {}
    
    # Add callback to update historical data with hidden symbol input
    @app.callback(
        Output("historical-chart", "figure", allow_duplicate=True),
        [Input("load-historical-data", "n_clicks")],
        [State("hidden-symbol-input", "value"),
         State("historical-period", "value"),
         State("historical-frequency", "value")],
        prevent_initial_call=True
    )
    def update_historical_data_from_hidden(n_clicks, symbol, period, frequency):
        """Update historical data chart using hidden symbol input"""
        if not n_clicks or not symbol:
            return dash.no_update
            
        try:
            # Map period and frequency to API parameters
            period_mapping = {
                'day': ('day', 1, 'minute', 5),
                'week': ('day', 7, 'minute', 30),
                'month': ('month', 1, 'daily', 1),
                '3month': ('month', 3, 'daily', 1),
                'year': ('year', 1, 'daily', 1)
            }
            
            # Get period parameters
            period_type, period_value, freq_type, freq_value = period_mapping.get(period, ('month', 1, 'daily', 1))
            
            # Override frequency if specified
            if frequency:
                freq_mapping = {
                    '1min': ('minute', 1),
                    '5min': ('minute', 5),
                    '15min': ('minute', 15),
                    '30min': ('minute', 30),
                    '60min': ('minute', 60),
                    'daily': ('daily', 1)
                }
                if frequency in freq_mapping:
                    freq_type, freq_value = freq_mapping[frequency]
            
            # Get historical data
            data_collector = DataCollector()
            historical_data = data_collector.get_historical_data(
                symbol, 
                period_type=period_type, 
                period=period_value, 
                frequency_type=freq_type, 
                frequency=freq_value
            )
            
            # Create candlestick chart
            if historical_data is not None and not historical_data.empty:
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=historical_data.index if hasattr(historical_data, 'index') else historical_data['datetime'],
                        open=historical_data['open'],
                        high=historical_data['high'],
                        low=historical_data['low'],
                        close=historical_data['close'],
                        name="Price"
                    )
                ])
                
                # Add volume as bar chart on secondary y-axis
                fig.add_trace(
                    go.Bar(
                        x=historical_data.index if hasattr(historical_data, 'index') else historical_data['datetime'],
                        y=historical_data['volume'],
                        name="Volume",
                        yaxis="y2",
                        marker_color='rgba(0, 0, 255, 0.5)'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Historical Data for {symbol}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    ),
                    height=600
                )
                
                return fig
            else:
                return go.Figure()
                
        except Exception as e:
            logger.error(f"Error updating historical chart: {str(e)}")
            return go.Figure()
    
    logger.info("Callbacks registered successfully")

def test_recommendation_generation(recommendation_engine, symbol="AAPL", timeframe="30m"):
    """
    Test recommendation generation functionality
    
    Args:
        recommendation_engine: ShortTermRecommendationEngine instance
        symbol (str): Symbol to test with
        timeframe (str): Timeframe to test with
        
    Returns:
        bool: True if test passed, False otherwise
    """
    logger.info(f"Testing recommendation generation for {symbol} ({timeframe})...")
    
    try:
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(
            symbol=symbol,
            timeframe=timeframe,
            option_type="ALL",
            min_confidence=0.6,
            min_risk_reward=1.5
        )
        
        # Check if recommendations were generated
        if recommendations is None:
            logger.error("Recommendations is None")
            return False
            
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        # Check recommendation structure
        if len(recommendations) > 0:
            rec = recommendations[0]
            required_fields = [
                'symbol', 'option_type', 'strike', 'expiration', 'confidence_score',
                'risk_reward_ratio', 'potential_return'
            ]
            
            for field in required_fields:
                if field not in rec:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            logger.info("Recommendation structure is valid")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing recommendation generation: {str(e)}")
        return False

def test_validation_visualization(recommendation_engine, data_pipeline, symbol="AAPL", timeframe="30m"):
    """
    Test validation visualization functionality
    
    Args:
        recommendation_engine: ShortTermRecommendationEngine instance
        data_pipeline: ShortTermDataPipeline instance
        symbol (str): Symbol to test with
        timeframe (str): Timeframe to test with
        
    Returns:
        bool: True if test passed, False otherwise
    """
    logger.info(f"Testing validation visualization for {symbol} ({timeframe})...")
    
    try:
        # Generate a recommendation
        recommendations = recommendation_engine.generate_recommendations(
            symbol=symbol,
            timeframe=timeframe,
            option_type="ALL",
            min_confidence=0.6,
            min_risk_reward=1.5
        )
        
        if not recommendations:
            logger.error("No recommendations generated for testing")
            return False
            
        # Get validation data
        validation_data = recommendation_engine.get_validation_data(recommendations[0], timeframe)
        
        if not validation_data:
            logger.error("No validation data generated")
            return False
            
        # Check validation data structure
        required_sections = ['price_data', 'key_levels', 'indicators', 'recommendation']
        
        for section in required_sections:
            if section not in validation_data:
                logger.error(f"Missing required section in validation data: {section}")
                return False
        
        logger.info("Validation data structure is valid")
        
        # Test creating validation chart
        from app.visualizations.validation_charts import create_validation_chart
        chart = create_validation_chart(validation_data)
        
        if not chart:
            logger.error("Failed to create validation chart")
            return False
            
        logger.info("Validation chart created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing validation visualization: {str(e)}")
        return False

def run_all_tests(recommendation_engine, data_pipeline):
    """
    Run all tests for the implementation
    
    Args:
        recommendation_engine: ShortTermRecommendationEngine instance
        data_pipeline: ShortTermDataPipeline instance
        
    Returns:
        dict: Test results
    """
    logger.info("Running all tests...")
    
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    test_timeframes = ["15m", "30m", "60m", "120m"]
    
    results = {
        "recommendation_generation": {},
        "validation_visualization": {}
    }
    
    # Test recommendation generation
    for symbol in test_symbols:
        for timeframe in test_timeframes:
            test_key = f"{symbol}_{timeframe}"
            results["recommendation_generation"][test_key] = test_recommendation_generation(
                recommendation_engine, symbol, timeframe
            )
    
    # Test validation visualization
    for symbol in test_symbols[:1]:  # Just test one symbol for visualization
        for timeframe in test_timeframes:
            test_key = f"{symbol}_{timeframe}"
            results["validation_visualization"][test_key] = test_validation_visualization(
                recommendation_engine, data_pipeline, symbol, timeframe
            )
    
    # Calculate overall results
    recommendation_success = sum(results["recommendation_generation"].values()) / len(results["recommendation_generation"])
    visualization_success = sum(results["validation_visualization"].values()) / len(results["validation_visualization"])
    
    overall_success = (recommendation_success + visualization_success) / 2
    
    results["summary"] = {
        "recommendation_success_rate": recommendation_success,
        "visualization_success_rate": visualization_success,
        "overall_success_rate": overall_success,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Tests completed. Overall success rate: {overall_success:.2%}")
    
    return results
