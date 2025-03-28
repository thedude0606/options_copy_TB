"""
Exit Strategy Visualization Component

This module provides visualization components for displaying exit strategy recommendations
in the dashboard, showing when to sell options and at what premium.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ExitStrategyDisplay:
    """
    Component for displaying exit strategy recommendations in the dashboard.
    """
    
    @staticmethod
    def create_exit_strategy_card(recommendation):
        """
        Create a card displaying exit strategy details for a recommendation.
        
        Args:
            recommendation (dict): Recommendation with exit strategy
            
        Returns:
            dash component: Card with exit strategy details
        """
        # Extract exit strategy if available
        exit_strategy = recommendation.get('exitStrategy', {})
        if not exit_strategy:
            return html.Div("No exit strategy available")
        
        # Extract key information
        symbol = recommendation.get('symbol', 'Unknown')
        underlying = recommendation.get('underlying', 'Unknown')
        option_type = recommendation.get('option_type', 'CALL')
        strike = recommendation.get('strike', 0)
        expiration_date = recommendation.get('expiration_date', 'Unknown')
        entry_price = recommendation.get('price', 0)
        
        # Format exit strategy information
        optimal_exit_date = exit_strategy.get('optimalExitDate', 'Unknown')
        days_to_hold = exit_strategy.get('daysToHold', 0)
        stop_loss = exit_strategy.get('stopLoss', 0)
        take_profit = exit_strategy.get('takeProfit', 0)
        exit_probability = exit_strategy.get('exitProbability', 0)
        confidence_score = exit_strategy.get('confidenceScore', 0)
        
        # Format dates
        if isinstance(optimal_exit_date, str):
            try:
                optimal_exit_date = datetime.fromisoformat(optimal_exit_date.replace('Z', '+00:00'))
                optimal_exit_date = optimal_exit_date.strftime('%Y-%m-%d')
            except:
                pass
        
        # Create price targets table
        price_targets = exit_strategy.get('priceTargets', [])
        price_targets_table = html.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Target Price"),
                        html.Th("% of Position"),
                        html.Th("Profit %")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(f"${target.get('price', 0):.2f}"),
                        html.Td(f"{target.get('percentage', 0) * 100:.0f}%"),
                        html.Td(f"{target.get('profit_percentage', 0):.1f}%")
                    ]) for target in price_targets
                ])
            ],
            className="table table-sm table-striped"
        ) if price_targets else html.Div("No price targets available")
        
        # Create exit reasons list
        exit_reasons = exit_strategy.get('exitReasons', [])
        exit_reasons_list = html.Ul([
            html.Li(reason) for reason in exit_reasons
        ]) if exit_reasons else html.Div("No exit reasons available")
        
        # Create card
        card = dbc.Card(
            dbc.CardBody([
                html.H5(f"Exit Strategy for {symbol} {option_type} {strike} {expiration_date}", className="card-title"),
                
                html.Div([
                    html.Div([
                        html.Strong("Optimal Exit Date:"),
                        html.Span(f" {optimal_exit_date}", className="ms-2")
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Strong("Days to Hold:"),
                        html.Span(f" {days_to_hold}", className="ms-2")
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Strong("Stop Loss:"),
                        html.Span(f" ${stop_loss:.2f}", className="ms-2")
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Strong("Take Profit:"),
                        html.Span(f" ${take_profit:.2f}", className="ms-2")
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Strong("Exit Probability:"),
                        html.Span(f" {exit_probability * 100:.1f}%", className="ms-2")
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Strong("Confidence Score:"),
                        html.Span(f" {confidence_score * 100:.1f}%", className="ms-2")
                    ], className="mb-2"),
                ], className="mb-3"),
                
                html.H6("Price Targets", className="mt-3"),
                price_targets_table,
                
                html.H6("Exit Reasons", className="mt-3"),
                exit_reasons_list,
                
                # Add visualization
                dcc.Graph(
                    figure=ExitStrategyDisplay.create_exit_strategy_chart(recommendation),
                    config={'displayModeBar': False}
                )
            ])
        )
        
        return card
    
    @staticmethod
    def create_exit_strategy_chart(recommendation):
        """
        Create a chart visualizing the exit strategy.
        
        Args:
            recommendation (dict): Recommendation with exit strategy
            
        Returns:
            plotly.graph_objects.Figure: Chart visualizing exit strategy
        """
        # Extract exit strategy if available
        exit_strategy = recommendation.get('exitStrategy', {})
        if not exit_strategy:
            # Return empty figure
            return go.Figure()
        
        # Extract key information
        entry_price = recommendation.get('price', 0)
        stop_loss = exit_strategy.get('stopLoss', 0)
        take_profit = exit_strategy.get('takeProfit', 0)
        
        # Extract price targets
        price_targets = exit_strategy.get('priceTargets', [])
        
        # Create time points (days from now)
        days_to_hold = exit_strategy.get('daysToHold', 7)
        time_points = list(range(days_to_hold + 1))
        
        # Create price scenarios
        # Optimistic: Entry to take profit
        # Expected: Entry to highest probability target
        # Pessimistic: Entry to stop loss
        
        # Find expected target (highest probability or middle target)
        if price_targets:
            # Sort by percentage (highest first)
            sorted_targets = sorted(price_targets, key=lambda x: x.get('percentage', 0), reverse=True)
            expected_target = sorted_targets[0].get('price', entry_price * 1.2)
        else:
            expected_target = (entry_price + take_profit) / 2
        
        # Create price paths
        optimistic_path = np.linspace(entry_price, take_profit, days_to_hold + 1)
        expected_path = np.linspace(entry_price, expected_target, days_to_hold + 1)
        pessimistic_path = np.linspace(entry_price, stop_loss, days_to_hold + 1)
        
        # Create figure
        fig = go.Figure()
        
        # Add price paths
        fig.add_trace(go.Scatter(
            x=time_points,
            y=optimistic_path,
            mode='lines',
            name='Optimistic',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=expected_path,
            mode='lines',
            name='Expected',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=pessimistic_path,
            mode='lines',
            name='Pessimistic',
            line=dict(color='red', width=2)
        ))
        
        # Add entry point
        fig.add_trace(go.Scatter(
            x=[0],
            y=[entry_price],
            mode='markers',
            name='Entry',
            marker=dict(color='blue', size=10)
        ))
        
        # Add exit points
        for i, target in enumerate(price_targets):
            target_price = target.get('price', 0)
            percentage = target.get('percentage', 0)
            profit_pct = target.get('profit_percentage', 0)
            
            # Calculate x position (distribute targets across time range)
            x_pos = days_to_hold * (i + 1) / (len(price_targets) + 1)
            
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[target_price],
                mode='markers',
                name=f'Target {i+1}: {profit_pct:.1f}%',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
        
        # Add stop loss and take profit lines
        fig.add_shape(
            type="line",
            x0=0,
            y0=stop_loss,
            x1=days_to_hold,
            y1=stop_loss,
            line=dict(color="red", width=1, dash="dash"),
            name="Stop Loss"
        )
        
        fig.add_shape(
            type="line",
            x0=0,
            y0=take_profit,
            x1=days_to_hold,
            y1=take_profit,
            line=dict(color="green", width=1, dash="dash"),
            name="Take Profit"
        )
        
        # Add annotations
        fig.add_annotation(
            x=days_to_hold,
            y=stop_loss,
            text=f"Stop Loss: ${stop_loss:.2f}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom"
        )
        
        fig.add_annotation(
            x=days_to_hold,
            y=take_profit,
            text=f"Take Profit: ${take_profit:.2f}",
            showarrow=False,
            xanchor="right",
            yanchor="top"
        )
        
        # Update layout
        fig.update_layout(
            title="Exit Strategy Visualization",
            xaxis_title="Days from Entry",
            yaxis_title="Option Price ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        
        return fig

def create_exit_strategy_tab():
    """
    Create a tab for displaying exit strategies for current positions.
    
    Returns:
        dash component: Tab content
    """
    return html.Div([
        html.H3("Exit Strategy Recommendations", className="mb-4"),
        
        html.Div([
            html.P("This tab displays exit strategy recommendations for your current options positions, "
                   "including when to sell and at what premium.", className="lead")
        ], className="mb-4"),
        
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Position Details"),
                        dbc.CardBody([
                            dbc.InputGroup([
                                dbc.InputGroupText("Symbol"),
                                dbc.Input(id="exit-strategy-symbol-input", placeholder="Enter option symbol")
                            ], className="mb-3"),
                            
                            dbc.InputGroup([
                                dbc.InputGroupText("Entry Price"),
                                dbc.Input(id="exit-strategy-price-input", type="number", placeholder="Enter entry price")
                            ], className="mb-3"),
                            
                            dbc.InputGroup([
                                dbc.InputGroupText("Entry Date"),
                                dbc.Input(id="exit-strategy-date-input", type="date")
                            ], className="mb-3"),
                            
                            dbc.Button("Generate Exit Strategy", id="generate-exit-strategy-button", color="primary", className="w-100")
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    html.Div(id="exit-strategy-output")
                ], width=8)
            ])
        ]),
        
        html.Hr(),
        
        html.H4("Current Positions with Exit Strategies", className="mt-4 mb-3"),
        html.Div(id="positions-exit-strategies")
    ])

def register_exit_strategy_callbacks(app, recommendation_engine):
    """
    Register callbacks for exit strategy tab.
    
    Args:
        app: Dash app instance
        recommendation_engine: Recommendation engine instance
    """
    @app.callback(
        dash.Output("exit-strategy-output", "children"),
        [dash.Input("generate-exit-strategy-button", "n_clicks")],
        [dash.State("exit-strategy-symbol-input", "value"),
         dash.State("exit-strategy-price-input", "value"),
         dash.State("exit-strategy-date-input", "value")]
    )
    def generate_exit_strategy(n_clicks, symbol, price, date):
        """
        Generate exit strategy for a position.
        
        Args:
            n_clicks: Button clicks
            symbol: Option symbol
            price: Entry price
            date: Entry date
            
        Returns:
            dash component: Exit strategy card
        """
        if n_clicks is None or not symbol or not price:
            return html.Div("Enter position details and click Generate Exit Strategy")
        
        # Parse entry date
        if date:
            entry_date = datetime.strptime(date, '%Y-%m-%d')
        else:
            entry_date = datetime.now()
        
        # Create position data
        position_data = {
            'symbol': symbol,
            'entry_price': float(price),
            'entry_date': entry_date,
            'position_type': 'long',
            'option_data': {
                'symbol': symbol,
                'option_type': 'CALL' if 'C' in symbol else 'PUT',
                'implied_volatility': 0.3,  # Default value
                'delta': 0.5,  # Default value
                'gamma': 0.05,  # Default value
                'theta': -0.05,  # Default value
                'vega': 0.1,  # Default value
                'rho': 0.01  # Default value
            }
        }
        
        # Generate exit strategy
        exit_strategy = recommendation_engine.generate_exit_strategy_for_position(position_data)
        
        # Create recommendation with exit strategy
        recommendation = {
            'symbol': symbol,
            'price': float(price),
            'exitStrategy': {
                'optimalExitDate': exit_strategy['optimal_exit_time'],
                'daysToHold': exit_strategy['days_to_hold'],
                'priceTargets': exit_strategy['price_targets'],
                'stopLoss': exit_strategy['stop_loss'],
                'takeProfit': exit_strategy['take_profit'],
                'exitProbability': exit_strategy['exit_probability'],
                'exitReasons': exit_strategy['exit_reasons'],
                'confidenceScore': exit_strategy['confidence_score']
            }
        }
        
        # Create exit strategy card
        return ExitStrategyDisplay.create_exit_strategy_card(recommendation)
    
    @app.callback(
        dash.Output("positions-exit-strategies", "children"),
        [dash.Input("interval-component", "n_intervals")]
    )
    def update_positions_exit_strategies(n_intervals):
        """
        Update exit strategies for current positions.
        
        Args:
            n_intervals: Interval component trigger
            
        Returns:
            dash component: Cards with exit strategies for current positions
        """
        # This would typically fetch current positions from a database or API
        # For demonstration, we'll create some sample positions
        sample_positions = [
            {
                'symbol': 'AAPL220121C00150000',
                'underlying': 'AAPL',
                'option_type': 'CALL',
                'strike': 150,
                'expiration_date': '2022-01-21',
                'price': 5.25,
                'exitStrategy': {
                    'optimalExitDate': (datetime.now() + timedelta(days=5)).isoformat(),
                    'daysToHold': 5,
                    'priceTargets': [
                        {'price': 6.30, 'percentage': 0.5, 'profit_percentage': 20},
                        {'price': 7.35, 'percentage': 0.5, 'profit_percentage': 40}
                    ],
                    'stopLoss': 4.20,
                    'takeProfit': 7.35,
                    'exitProbability': 0.65,
                    'exitReasons': ['Moderate time decay', 'High implied volatility'],
                    'confidenceScore': 0.7
                }
            },
            {
                'symbol': 'SPY220218P00440000',
                'underlying': 'SPY',
                'option_type': 'PUT',
                'strike': 440,
                'expiration_date': '2022-02-18',
                'price': 8.75,
                'exitStrategy': {
                    'optimalExitDate': (datetime.now() + timedelta(days=10)).isoformat(),
                    'daysToHold': 10,
                    'priceTargets': [
                        {'price': 10.50, 'percentage': 0.3, 'profit_percentage': 20},
                        {'price': 11.40, 'percentage': 0.3, 'profit_percentage': 30},
                        {'price': 13.10, 'percentage': 0.4, 'profit_percentage': 50}
                    ],
                    'stopLoss': 7.00,
                    'takeProfit': 13.10,
                    'exitProbability': 0.55,
                    'exitReasons': ['Low time decay', 'Market downtrend'],
                    'confidenceScore': 0.6
                }
            }
        ]
        
        # Create cards for each position
        cards = [
            dbc.Col(ExitStrategyDisplay.create_exit_strategy_card(position), width=6, className="mb-4")
            for position in sample_positions
        ]
        
        # Arrange cards in rows
        rows = []
        for i in range(0, len(cards), 2):
            row_cards = cards[i:i+2]
            rows.append(dbc.Row(row_cards, className="mb-4"))
        
        return html.Div(rows)
