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
                
            # Use the first recommendation for visualization
            recommendation = recommendations[0]
            
            # Create validation visualization
            validation_chart = create_validation_chart(
                symbol=symbol,
                option_type=recommendation["option_type"],
                strike_price=recommendation["strike_price"],
                expiration_date=recommendation["expiration_date"],
                timeframe=timeframe or "30m"
            )
            
            # Create comparison chart
            comparison_chart = create_timeframe_comparison_chart(
                symbol=symbol,
                timeframes=["15m", "30m", "60m", "120m"]
            )
            
            # Create sentiment chart
            sentiment_chart = create_sentiment_chart(symbol=symbol)
            
            # Create risk/reward visualization
            risk_reward_viz = create_risk_reward_visualization(
                potential_return=recommendation["potential_return"],
                risk_reward=recommendation["risk_reward"],
                confidence=recommendation["confidence"]
            )
            
            # Combine all visualizations
            return html.Div([
                html.H4("Validation Details"),
                html.Div([
                    html.Div([
                        html.H5("Price Action & Entry/Exit Points"),
                        validation_chart
                    ], className="col-md-6"),
                    html.Div([
                        html.H5("Timeframe Comparison"),
                        comparison_chart
                    ], className="col-md-6")
                ], className="row"),
                html.Div([
                    html.Div([
                        html.H5("Sentiment Analysis"),
                        sentiment_chart
                    ], className="col-md-6"),
                    html.Div([
                        html.H5("Risk/Reward Profile"),
                        risk_reward_viz
                    ], className="col-md-6")
                ], className="row mt-4")
            ])
            
        except Exception as e:
            logger.error(f"Error creating validation visualization: {str(e)}")
            return html.Div(f"Error: {str(e)}")
    
    # Callback for market overview updates
    @app.callback(
        Output("market-overview-table", "children"),
        [Input("interval-component", "n_intervals")]
    )
    def update_market_overview(n_intervals):
        """Update market overview table with latest data"""
        try:
            # Get market overview data
            market_data = data_pipeline.get_market_overview()
            
            if not market_data:
                return html.Div("No market data available")
                
            # Create table rows
            rows = []
            for symbol, data in market_data.items():
                if not data:
                    continue
                    
                # Extract values
                last_price = data.get('last_price', 0)
                change = data.get('change', 0)
                change_percent = data.get('change_percent', 0)
                
                # Format change with color
                if change > 0:
                    change_text = html.Span(f"+{change:.2f} (+{change_percent:.2f}%)", style={"color": "green"})
                elif change < 0:
                    change_text = html.Span(f"{change:.2f} ({change_percent:.2f}%)", style={"color": "red"})
                else:
                    change_text = html.Span(f"{change:.2f} ({change_percent:.2f}%)")
                
                # Create row
                row = html.Tr([
                    html.Td(symbol),
                    html.Td(f"${last_price:.2f}"),
                    html.Td(change_text)
                ])
                
                rows.append(row)
            
            # Create table
            table = html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Index"),
                        html.Th("Price"),
                        html.Th("Change")
                    ])
                ),
                html.Tbody(rows)
            ], className="table table-striped")
            
            return table
            
        except Exception as e:
            logger.error(f"Error updating market overview: {str(e)}")
            return html.Div(f"Error: {str(e)}")
    
    # Callback for timeframe selection
    @app.callback(
        Output("apply-timeframe", "children"),
        [Input("trading-timeframe", "value")]
    )
    def update_timeframe_button(timeframe):
        """Update timeframe button text"""
        return f"Apply {timeframe or '30m'} Timeframe"
    
    # Callback for feature tab content
    @app.callback(
        Output("feature-content", "children"),
        [Input("feature-tabs", "value")]
    )
    def update_feature_content(tab):
        """Update feature tab content based on selected tab"""
        if tab == "options-chain":
            try:
                return create_options_chain_tab()
            except Exception as e:
                logger.error(f"Error displaying options chain: {str(e)}")
                return html.Div(f"Error loading Options Chain: {str(e)}")
                
        elif tab == "greeks":
            try:
                return create_greeks_tab()
            except Exception as e:
                logger.error(f"Error displaying Greeks: {str(e)}")
                return html.Div(f"Error loading Greeks: {str(e)}")
                
        elif tab == "indicators":
            try:
                return create_indicators_tab()
            except Exception as e:
                logger.error(f"Error displaying indicators: {str(e)}")
                return html.Div(f"Error loading Technical Indicators: {str(e)}")
                
        elif tab == "historical":
            try:
                return create_historical_tab()
            except Exception as e:
                logger.error(f"Error displaying historical data: {str(e)}")
                return html.Div(f"Error loading Historical Data: {str(e)}")
                
        elif tab == "real-time":
            try:
                return get_real_time_tab_layout()
            except Exception as e:
                logger.error(f"Error displaying real-time data: {str(e)}")
                return html.Div(f"Error loading Real-Time Data: {str(e)}")
                
        return html.Div([
            html.P("Select a tab to view content.")
        ])
    
    # Add callback to update real-time data
    @app.callback(
        Output("rt-stream-data", "data", allow_duplicate=True),
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
            # This is a simplified implementation that would be replaced with actual streaming data
            # In a real implementation, this would connect to the Schwab API streaming service
            
            # Create dummy data for demonstration
            data = {}
            for symbol in symbols:
                # Generate random price data
                base_price = 100 + (hash(symbol) % 400)  # Different base price for each symbol
                current_time = datetime.now().isoformat()
                
                if data_type == "quotes":
                    # Quote data
                    data[symbol] = {
                        "last_price": base_price + (n_intervals % 10) * 0.25,
                        "bid": base_price + (n_intervals % 10) * 0.25 - 0.05,
                        "ask": base_price + (n_intervals % 10) * 0.25 + 0.05,
                        "volume": 1000 + (n_intervals * 100),
                        "timestamp": current_time
                    }
                else:
                    # Options data
                    data[symbol] = {
                        "calls": [
                            {
                                "strike": base_price - 5,
                                "last": 5.25 + (n_intervals % 5) * 0.1,
                                "bid": 5.20 + (n_intervals % 5) * 0.1,
                                "ask": 5.30 + (n_intervals % 5) * 0.1,
                                "volume": 500 + (n_intervals * 50),
                                "open_interest": 2000,
                                "timestamp": current_time
                            },
                            {
                                "strike": base_price,
                                "last": 2.50 + (n_intervals % 5) * 0.1,
                                "bid": 2.45 + (n_intervals % 5) * 0.1,
                                "ask": 2.55 + (n_intervals % 5) * 0.1,
                                "volume": 800 + (n_intervals * 50),
                                "open_interest": 3500,
                                "timestamp": current_time
                            },
                            {
                                "strike": base_price + 5,
                                "last": 1.25 + (n_intervals % 5) * 0.05,
                                "bid": 1.20 + (n_intervals % 5) * 0.05,
                                "ask": 1.30 + (n_intervals % 5) * 0.05,
                                "volume": 600 + (n_intervals * 50),
                                "open_interest": 2500,
                                "timestamp": current_time
                            }
                        ],
                        "puts": [
                            {
                                "strike": base_price - 5,
                                "last": 1.15 + (n_intervals % 5) * 0.05,
                                "bid": 1.10 + (n_intervals % 5) * 0.05,
                                "ask": 1.20 + (n_intervals % 5) * 0.05,
                                "volume": 450 + (n_intervals * 40),
                                "open_interest": 1800,
                                "timestamp": current_time
                            },
                            {
                                "strike": base_price,
                                "last": 2.40 + (n_intervals % 5) * 0.1,
                                "bid": 2.35 + (n_intervals % 5) * 0.1,
                                "ask": 2.45 + (n_intervals % 5) * 0.1,
                                "volume": 750 + (n_intervals * 40),
                                "open_interest": 3200,
                                "timestamp": current_time
                            },
                            {
                                "strike": base_price + 5,
                                "last": 5.10 + (n_intervals % 5) * 0.1,
                                "bid": 5.05 + (n_intervals % 5) * 0.1,
                                "ask": 5.15 + (n_intervals % 5) * 0.1,
                                "volume": 550 + (n_intervals * 40),
                                "open_interest": 2300,
                                "timestamp": current_time
                            }
                        ]
                    }
            
            return data
            
        except Exception as e:
            logger.error(f"Error updating stream data: {str(e)}")
            return {}
    
    # Callback for price chart
    @app.callback(
        Output("rt-price-chart", "figure"),
        [Input("rt-stream-data", "data")]
    )
    def update_price_chart(stream_data):
        """Update price chart with streaming data"""
        # Create empty figure
        fig = go.Figure()
        
        # Check if we have data
        if not stream_data:
            fig.update_layout(
                title="No streaming data available",
                xaxis_title="Time",
                yaxis_title="Price",
                height=400
            )
            return fig
        
        try:
            # Process each symbol
            for symbol, data in stream_data.items():
                if "last_price" in data:  # Quote data
                    # In a real implementation, we would accumulate data points over time
                    # For demonstration, we'll just plot the current point
                    fig.add_trace(go.Scatter(
                        x=[datetime.now().strftime("%H:%M:%S")],
                        y=[data["last_price"]],
                        mode="markers+lines",
                        name=symbol
                    ))
            
            # Update layout
            fig.update_layout(
                title="Real-Time Price Data",
                xaxis_title="Time",
                yaxis_title="Price",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error updating price chart: {str(e)}")
            fig.update_layout(
                title=f"Error: {str(e)}",
                xaxis_title="Time",
                yaxis_title="Price",
                height=400
            )
            return fig
    
    # Callback for data table
    @app.callback(
        Output("rt-data-table", "children"),
        [Input("rt-stream-data", "data")]
    )
    def update_data_table(stream_data):
        """Update data table with streaming data"""
        if not stream_data:
            return html.Div("No streaming data available")
        
        try:
            # Create table rows
            rows = []
            for symbol, data in stream_data.items():
                if "last_price" in data:  # Quote data
                    # Create row
                    row = html.Tr([
                        html.Td(symbol),
                        html.Td(f"${data['last_price']:.2f}"),
                        html.Td(f"${data['bid']:.2f}"),
                        html.Td(f"${data['ask']:.2f}"),
                        html.Td(f"{data['volume']:,}"),
                        html.Td(data['timestamp'])
                    ])
                    rows.append(row)
            
            if not rows:
                return html.Div("No quote data available")
            
            # Create table
            table = html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Last"),
                        html.Th("Bid"),
                        html.Th("Ask"),
                        html.Th("Volume"),
                        html.Th("Timestamp")
                    ])
                ),
                html.Tbody(rows)
            ], className="table table-striped")
            
            return table
            
        except Exception as e:
            logger.error(f"Error updating data table: {str(e)}")
            return html.Div(f"Error: {str(e)}")
    
    # Callback for time & sales
    @app.callback(
        Output("rt-time-sales", "children"),
        [Input("rt-stream-data", "data")]
    )
    def update_time_sales(stream_data):
        """Update time & sales with streaming data"""
        if not stream_data:
            return html.Div("No streaming data available")
        
        try:
            # In a real implementation, this would show individual trades
            # For demonstration, we'll just show the current data
            
            # Create content
            content = html.Div([
                html.P("Time & Sales data would be displayed here."),
                html.P("This would typically show individual trades as they occur."),
                html.Pre(json.dumps(stream_data, indent=2))
            ])
            
            return content
            
        except Exception as e:
            logger.error(f"Error updating time & sales: {str(e)}")
            return html.Div(f"Error: {str(e)}")
    
    logger.info("Callbacks registered successfully")

def create_options_chain_tab():
    """
    Create the options chain tab
    
    Returns:
        html.Div: The options chain tab component
    """
    return html.Div([
        html.H3("Options Chain"),
        
        # Symbol and expiration selection
        html.Div([
            html.Div([
                html.Label("Symbol:"),
                dcc.Input(
                    id="oc-symbol-input",
                    type="text",
                    value="",
                    placeholder="Enter symbol (e.g., AAPL)",
                    style={"width": "150px", "margin-right": "10px"}
                ),
            ], className="col-md-3"),
            
            html.Div([
                html.Label("Expiration:"),
                dcc.Dropdown(
                    id="oc-expiration-dropdown",
                    options=[],
                    value=None,
                    placeholder="Select expiration date",
                    style={"width": "100%"}
                ),
            ], className="col-md-3"),
            
            html.Div([
                html.Button(
                    "Load Chain",
                    id="oc-load-button",
                    n_clicks=0,
                    style={"margin-top": "24px"}
                ),
            ], className="col-md-2"),
        ], className="row mb-3"),
        
        # Options chain display
        html.Div([
            # Calls
            html.Div([
                html.H4("Calls"),
                html.Div(id="oc-calls-table")
            ], className="col-md-6"),
            
            # Puts
            html.Div([
                html.H4("Puts"),
                html.Div(id="oc-puts-table")
            ], className="col-md-6"),
        ], className="row"),
        
        # Hidden divs for storing data
        dcc.Store(id="oc-data-store"),
    ])

def test_platform(data_collector):
    """
    Test the options recommendation platform
    
    Args:
        data_collector: DataCollector instance
        
    Returns:
        bool: True if tests pass, False otherwise
    """
    logger.info("Testing platform...")
    
    try:
        # Initialize components
        data_pipeline = ShortTermDataPipeline(data_collector)
        recommendation_engine = ShortTermRecommendationEngine(data_pipeline)
        
        # Test market data retrieval
        market_data = data_pipeline.get_market_overview()
        if not market_data:
            logger.error("Failed to retrieve market data")
            return False
        logger.info(f"Successfully retrieved market data: {market_data}")
        
        # Test recommendation generation
        symbol = "SPY"
        recommendations = recommendation_engine.generate_recommendations(
            symbol=symbol,
            timeframe="30m",
            option_type="ALL",
            min_confidence=0.6,
            min_risk_reward=1.5
        )
        if not recommendations:
            logger.warning(f"No recommendations generated for {symbol}")
        else:
            logger.info(f"Successfully generated {len(recommendations)} recommendations for {symbol}")
        
        return True
        
    except Exception as e:
        logger.error(f"Platform test failed: {str(e)}")
        return False
