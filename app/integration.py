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
            try:
                # Check if required keys exist in recommendation
                required_keys = ["option_type", "strike_price", "expiration_date"]
                if all(key in recommendation for key in required_keys):
                    validation_chart = create_validation_chart(
                        symbol=symbol,
                        option_type=recommendation["option_type"],
                        strike_price=recommendation["strike_price"],
                        expiration_date=recommendation["expiration_date"],
                        timeframe=timeframe or "30m"
                    )
                else:
                    # Create a placeholder chart if keys are missing
                    missing_keys = [key for key in required_keys if key not in recommendation]
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"Cannot create validation chart: Missing data ({', '.join(missing_keys)})",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    validation_chart = fig
            except Exception as e:
                logger.error(f"Error creating validation visualization: {str(e)}")
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error creating validation chart: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                validation_chart = fig
            
            # Create comparison chart
            comparison_chart = create_timeframe_comparison_chart(
                symbol=symbol,
                timeframes=["15m", "30m", "60m", "120m"]
            )
            
            # Create sentiment chart
            sentiment_chart = create_sentiment_chart(symbol=symbol)
            
            # Create risk/reward visualization
            risk_reward_viz = create_risk_reward_visualization(
                symbol=symbol,
                option_type=recommendation["option_type"],
                strike_price=recommendation["strike_price"],
                expiration_date=recommendation["expiration_date"]
            )
            
            # Combine all visualizations
            return html.Div([
                html.H4(f"Validation for {symbol} {recommendation['option_type']} {recommendation['strike_price']} {recommendation['expiration_date']}"),
                html.Div([
                    html.Div([validation_chart], className="col-md-6"),
                    html.Div([comparison_chart], className="col-md-6")
                ], className="row"),
                html.Div([
                    html.Div([sentiment_chart], className="col-md-6"),
                    html.Div([risk_reward_viz], className="col-md-6")
                ], className="row mt-3")
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
                price = data.get('last_price', 0)
                change = data.get('change', 0)
                change_percent = data.get('change_percent', 0)
                
                # Format change with color
                change_class = "text-success" if change >= 0 else "text-danger"
                change_sign = "+" if change >= 0 else ""
                
                row = html.Tr([
                    html.Td(symbol),
                    html.Td(f"${price:.2f}"),
                    html.Td(f"{change_sign}{change:.2f} ({change_sign}{change_percent:.2f}%)", className=change_class)
                ])
                rows.append(row)
                
            # Create table
            table = dbc.Table(
                [html.Thead(html.Tr([
                    html.Th("Index"),
                    html.Th("Price"),
                    html.Th("Change")
                ]))],
                bordered=False,
                hover=True,
                responsive=True,
                striped=True,
                className="mb-0"
            )
            
            # Add rows to table
            table.children.append(html.Tbody(rows))
            
            return table
            
        except Exception as e:
            logger.error(f"Error updating market overview: {str(e)}")
            return html.Div(f"Error: {str(e)}")
    
    # Callback for timeframe selection
    @app.callback(
        Output("timeframe-output", "children"),
        [Input("trading-timeframe", "value")]
    )
    def update_timeframe_output(timeframe):
        """Update timeframe output"""
        if not timeframe:
            return "Selected timeframe: 30 Minutes (default)"
            
        timeframe_map = {
            "15m": "15 Minutes",
            "30m": "30 Minutes",
            "60m": "60 Minutes",
            "120m": "120 Minutes"
        }
        
        return f"Selected timeframe: {timeframe_map.get(timeframe, timeframe)}"
    
    # Callback for feature tab content
    @app.callback(
        Output("feature-content", "children"),
        [Input("feature-tabs", "active_tab")],
        [State("symbol-input", "value")]
    )
    def update_feature_content(active_tab, symbol):
        """Update feature tab content based on selected tab"""
        if not active_tab:
            return html.Div("Select a feature tab to view content")
            
        if not symbol:
            return html.Div("Enter a symbol and click Search to view feature content")
            
        try:
            if active_tab == "tab-options-chain":
                try:
                    # Get options chain data
                    options_chain = data_pipeline.get_options_chain(symbol)
                    if not options_chain or not options_chain.get('calls') or not options_chain.get('puts'):
                        return html.Div("No options chain data available for this symbol")
                    
                    # Create DataTable for calls
                    calls_df = pd.DataFrame(options_chain['calls'])
                    calls_table = dash_table.DataTable(
                        id='calls-table',
                        columns=[{"name": col, "id": col} for col in calls_df.columns],
                        data=calls_df.to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        page_size=10
                    )
                    
                    # Create DataTable for puts
                    puts_df = pd.DataFrame(options_chain['puts'])
                    puts_table = dash_table.DataTable(
                        id='puts-table',
                        columns=[{"name": col, "id": col} for col in puts_df.columns],
                        data=puts_df.to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        page_size=10
                    )
                    
                    return html.Div([
                        html.H4(f"Options Chain for {symbol}"),
                        html.Div([
                            html.H5("Calls"),
                            calls_table,
                            html.H5("Puts", className="mt-4"),
                            puts_table
                        ])
                    ])
                except Exception as e:
                    logger.error(f"Error displaying options chain: {str(e)}")
                    return html.Div(f"Error retrieving options chain: {str(e)}")
                
            elif active_tab == "tab-greeks":
                # Return the Greeks tab content
                return create_greeks_tab(symbol)
                
            elif active_tab == "tab-indicators":
                # Return the Technical Indicators tab content
                return create_indicators_tab(symbol)
                
            elif active_tab == "tab-historical":
                # Return the Historical Data tab content
                return create_historical_tab(symbol)
                
            elif active_tab == "tab-real-time":
                # Return the Real-Time Data tab content
                return get_real_time_tab_layout(symbol)
                
            else:
                return html.Div(f"Content for {active_tab} not implemented yet")
                
        except Exception as e:
            logger.error(f"Error updating feature content: {str(e)}")
            return html.Div(f"Error: {str(e)}")
    
    # Callback for real-time data stream
    @app.callback(
        Output("rt-stream-data", "data", allow_duplicate=True),
        [Input("interval-component", "n_intervals")],
        [State("symbol-input", "value")],
        prevent_initial_call=True
    )
    def update_stream_data(n_intervals, symbol):
        """Update real-time data stream"""
        if not symbol:
            return {"symbols": [], "data": {}}
            
        try:
            # Get real-time data
            stream_data = data_pipeline.get_real_time_data(symbol)
            
            if not stream_data:
                return {"symbols": [symbol], "data": {}}
                
            return {"symbols": [symbol], "data": stream_data}
            
        except Exception as e:
            logger.error(f"Error updating stream data: {str(e)}")
            return {"symbols": [], "data": {}, "error": str(e)}

def run_all_tests(recommendation_engine, data_pipeline):
    """
    Run all tests for the options recommendation platform
    
    Args:
        recommendation_engine: ShortTermRecommendationEngine instance
        data_pipeline: ShortTermDataPipeline instance
        
    Returns:
        dict: Test results with summary statistics
    """
    logger.info("Running all tests...")
    
    test_results = {
        "data_pipeline_tests": {},
        "recommendation_engine_tests": {},
        "integration_tests": {},
        "summary": {}
    }
    
    # Test symbols
    test_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT']
    
    # Test data pipeline
    data_pipeline_success = 0
    data_pipeline_total = 0
    
    for symbol in test_symbols:
        try:
            # Test market data retrieval
            market_data = data_pipeline.get_market_data(symbol)
            test_results["data_pipeline_tests"][f"market_data_{symbol}"] = {
                "success": market_data is not None,
                "data": str(market_data)[:100] + "..." if market_data else None
            }
            data_pipeline_total += 1
            if market_data is not None:
                data_pipeline_success += 1
                
            # Test options chain retrieval
            options_chain = data_pipeline.get_options_chain(symbol)
            test_results["data_pipeline_tests"][f"options_chain_{symbol}"] = {
                "success": options_chain is not None and len(options_chain.get('calls', [])) > 0,
                "data": f"Found {len(options_chain.get('calls', []))} calls and {len(options_chain.get('puts', []))} puts" if options_chain else None
            }
            data_pipeline_total += 1
            if options_chain is not None and len(options_chain.get('calls', [])) > 0:
                data_pipeline_success += 1
                
            # Test historical data retrieval
            historical_data = data_pipeline.get_historical_data(symbol, period_type="month", period_value=1, freq_type="daily", freq_value=1)
            test_results["data_pipeline_tests"][f"historical_data_{symbol}"] = {
                "success": historical_data is not None and len(historical_data) > 0,
                "data": f"Found {len(historical_data)} historical data points" if historical_data is not None else None
            }
            data_pipeline_total += 1
            if historical_data is not None and len(historical_data) > 0:
                data_pipeline_success += 1
                
        except Exception as e:
            logger.error(f"Error in data pipeline test for {symbol}: {str(e)}")
            test_results["data_pipeline_tests"][f"error_{symbol}"] = {
                "success": False,
                "error": str(e)
            }
            data_pipeline_total += 1
    
    # Test recommendation engine
    recommendation_engine_success = 0
    recommendation_engine_total = 0
    
    for symbol in test_symbols:
        try:
            # Test recommendation generation
            recommendations = recommendation_engine.generate_recommendations(
                symbol=symbol,
                timeframe="30m",
                option_type="ALL",
                min_confidence=0.6,
                min_risk_reward=1.5
            )
            test_results["recommendation_engine_tests"][f"recommendations_{symbol}"] = {
                "success": recommendations is not None,
                "data": f"Found {len(recommendations)} recommendations" if recommendations else None
            }
            recommendation_engine_total += 1
            if recommendations is not None:
                recommendation_engine_success += 1
                
        except Exception as e:
            logger.error(f"Error in recommendation engine test for {symbol}: {str(e)}")
            test_results["recommendation_engine_tests"][f"error_{symbol}"] = {
                "success": False,
                "error": str(e)
            }
            recommendation_engine_total += 1
    
    # Test integration
    integration_success = 0
    integration_total = 0
    
    # Test market overview
    try:
        market_overview = data_pipeline.get_market_overview()
        test_results["integration_tests"]["market_overview"] = {
            "success": market_overview is not None and len(market_overview) > 0,
            "data": f"Found {len(market_overview)} market overview items" if market_overview else None
        }
        integration_total += 1
        if market_overview is not None and len(market_overview) > 0:
            integration_success += 1
    except Exception as e:
        logger.error(f"Error in integration test for market overview: {str(e)}")
        test_results["integration_tests"]["error_market_overview"] = {
            "success": False,
            "error": str(e)
        }
        integration_total += 1
    
    # Calculate success rates
    data_pipeline_success_rate = data_pipeline_success / data_pipeline_total if data_pipeline_total > 0 else 0
    recommendation_engine_success_rate = recommendation_engine_success / recommendation_engine_total if recommendation_engine_total > 0 else 0
    integration_success_rate = integration_success / integration_total if integration_total > 0 else 0
    
    total_success = data_pipeline_success + recommendation_engine_success + integration_success
    total_tests = data_pipeline_total + recommendation_engine_total + integration_total
    overall_success_rate = total_success / total_tests if total_tests > 0 else 0
    
    # Add summary to test results
    test_results["summary"] = {
        "data_pipeline_success_rate": data_pipeline_success_rate,
        "recommendation_engine_success_rate": recommendation_engine_success_rate,
        "integration_success_rate": integration_success_rate,
        "overall_success_rate": overall_success_rate,
        "total_tests": total_tests,
        "total_success": total_success
    }
    
    logger.info(f"Test results: {overall_success_rate:.2%} success rate ({total_success}/{total_tests})")
    
    return test_results
