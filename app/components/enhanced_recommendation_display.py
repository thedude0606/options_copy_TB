"""
Enhanced recommendation display components for the options recommendation platform.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

class EnhancedRecommendationDisplay:
    """
    Class to create enhanced recommendation display components
    """
    
    def __init__(self):
        """
        Initialize the enhanced recommendation display
        """
        self.color_map = {
            'very_high': '#28a745',  # Green
            'high': '#5cb85c',       # Light green
            'moderate': '#ffc107',   # Yellow
            'low': '#f0ad4e',        # Orange
            'very_low': '#dc3545',   # Red
            'unknown': '#6c757d'     # Gray
        }
    
    def create_recommendation_card(self, option_data, option_type='CALL'):
        """
        Create a recommendation card for an option
        
        Args:
            option_data (dict): Option data
            option_type (str): Option type (CALL or PUT)
            
        Returns:
            dash component: Recommendation card
        """
        # Determine card color based on option type
        card_color = '#28a745' if option_type == 'CALL' else '#dc3545'  # Green for calls, red for puts
        
        # Format expiration date
        exp_date = option_data.get('expirationDate', '')
        if exp_date:
            try:
                exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d')
                formatted_exp_date = exp_date_obj.strftime('Exp: %b %d, %Y')
            except:
                formatted_exp_date = f"Exp: {exp_date}"
        else:
            formatted_exp_date = "Exp: N/A"
        
        # Get confidence data
        confidence_score = option_data.get('confidenceScore', 0)
        confidence_level = option_data.get('confidenceLevel', 'unknown')
        confidence_color = self.color_map.get(confidence_level, '#6c757d')
        
        # Get profit data
        expected_return = option_data.get('expectedReturn', 0)
        win_rate = option_data.get('winRate', 0)
        risk_reward = option_data.get('riskRewardRatio', 0)
        
        # Create card header
        header = dbc.CardHeader(
            [
                html.Div(
                    [
                        html.H5(f"{option_data.get('symbol', '')}", className="card-title mb-0"),
                        html.Span(f"{option_type}", className="badge float-right", 
                                 style={"background-color": card_color, "color": "white"})
                    ],
                    className="d-flex justify-content-between align-items-center"
                )
            ],
            style={"background-color": card_color, "color": "white"}
        )
        
        # Create card body
        body = dbc.CardBody(
            [
                html.Div(
                    [
                        html.H4(f"${option_data.get('strikePrice', 0):.2f} Strike", className="card-subtitle mb-2"),
                        html.P(formatted_exp_date, className="card-text text-muted")
                    ],
                    className="mb-3"
                ),
                
                # Confidence section
                html.Div(
                    [
                        html.P("Confidence", className="mb-1 font-weight-bold"),
                        html.Div(
                            dbc.Progress(
                                value=confidence_score,
                                color=confidence_color,
                                className="mb-1",
                                style={"height": "10px"}
                            )
                        ),
                        html.P(f"{confidence_score:.0f}%", className="text-right mb-0 small")
                    ],
                    className="mb-3"
                ),
                
                # Potential Return section
                html.Div(
                    [
                        html.P("Potential Return", className="mb-1 font-weight-bold"),
                        html.H4(f"{expected_return:.1f}%", className="text-right mb-0",
                               style={"color": "#28a745" if expected_return > 0 else "#dc3545"})
                    ],
                    className="mb-3"
                ),
                
                # Risk/Reward section
                html.Div(
                    [
                        html.Div(
                            [
                                html.P("Risk/Reward", className="mb-0"),
                                html.P(f"{risk_reward:.2f}x", className="mb-0")
                            ],
                            className="col-6"
                        ),
                        html.Div(
                            [
                                html.P("Entry Price", className="mb-0"),
                                html.P(f"${option_data.get('entryPrice', 0):.2f}", className="mb-0")
                            ],
                            className="col-6"
                        )
                    ],
                    className="row mb-3"
                ),
                
                # Action buttons
                html.Div(
                    [
                        dbc.Button("View Details", color="light", className="mr-2", id=f"view-details-{option_data.get('symbol', '')}-{option_data.get('strikePrice', 0)}"),
                        dbc.Button("Trade Now", color="primary", id=f"trade-now-{option_data.get('symbol', '')}-{option_data.get('strikePrice', 0)}")
                    ],
                    className="d-flex justify-content-between"
                )
            ]
        )
        
        # Create card
        card = dbc.Card(
            [header, body],
            className="mb-4 recommendation-card",
            style={"border-color": card_color}
        )
        
        return card
    
    def create_recommendation_detail_modal(self, option_data, option_type='CALL'):
        """
        Create a detailed modal for an option recommendation
        
        Args:
            option_data (dict): Option data
            option_type (str): Option type (CALL or PUT)
            
        Returns:
            dash component: Modal with detailed recommendation information
        """
        # Determine color based on option type
        color = '#28a745' if option_type == 'CALL' else '#dc3545'  # Green for calls, red for puts
        
        # Format expiration date
        exp_date = option_data.get('expirationDate', '')
        if exp_date:
            try:
                exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d')
                formatted_exp_date = exp_date_obj.strftime('%b %d, %Y')
            except:
                formatted_exp_date = exp_date
        else:
            formatted_exp_date = "N/A"
        
        # Get signal details
        signal_details = option_data.get('signalDetails', [])
        
        # Create modal
        modal = dbc.Modal(
            [
                dbc.ModalHeader(
                    f"{option_data.get('symbol', '')} {option_type} ${option_data.get('strikePrice', 0):.2f}",
                    style={"background-color": color, "color": "white"}
                ),
                dbc.ModalBody(
                    [
                        # Option details section
                        html.Div(
                            [
                                html.H5("Option Details", className="border-bottom pb-2"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P("Strike Price:", className="font-weight-bold mb-0"),
                                                html.P(f"${option_data.get('strikePrice', 0):.2f}", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Expiration:", className="font-weight-bold mb-0"),
                                                html.P(formatted_exp_date, className="mb-0")
                                            ],
                                            className="col-md-4"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Days to Exp:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('daysToExpiration', 0)}", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        )
                                    ],
                                    className="row mb-3"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P("Entry Price:", className="font-weight-bold mb-0"),
                                                html.P(f"${option_data.get('entryPrice', 0):.2f}", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Bid/Ask:", className="font-weight-bold mb-0"),
                                                html.P(f"${option_data.get('bid', 0):.2f} / ${option_data.get('ask', 0):.2f}", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Volume/OI:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('volume', 0)} / {option_data.get('openInterest', 0)}", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        )
                                    ],
                                    className="row mb-3"
                                )
                            ],
                            className="mb-4"
                        ),
                        
                        # Recommendation section
                        html.Div(
                            [
                                html.H5("Recommendation", className="border-bottom pb-2"),
                                html.P(option_data.get('recommendation', 'No recommendation available.'), className="mb-3"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P("Confidence Score:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('confidenceScore', 0):.0f}%", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Expected Return:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('expectedReturn', 0):.1f}%", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Win Rate:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('winRate', 0) * 100:.0f}%", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        )
                                    ],
                                    className="row mb-3"
                                )
                            ],
                            className="mb-4"
                        ),
                        
                        # Greeks section
                        html.Div(
                            [
                                html.H5("Option Greeks", className="border-bottom pb-2"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P("Delta:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('delta', 0):.4f}", className="mb-0")
                                            ],
                                            className="col"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Gamma:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('gamma', 0):.4f}", className="mb-0")
                                            ],
                                            className="col"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Theta:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('theta', 0):.4f}", className="mb-0")
                                            ],
                                            className="col"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Vega:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('vega', 0):.4f}", className="mb-0")
                                            ],
                                            className="col"
                                        ),
                                        html.Div(
                                            [
                                                html.P("IV:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('impliedVolatility', 0) * 100:.1f}%", className="mb-0")
                                            ],
                                            className="col"
                                        )
                                    ],
                                    className="row mb-3"
                                )
                            ],
                            className="mb-4"
                        ),
                        
                        # Exit strategy section
                        html.Div(
                            [
                                html.H5("Exit Strategy", className="border-bottom pb-2"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P("Take Profit:", className="font-weight-bold mb-0"),
                                                html.P(f"${option_data.get('takeProfitPrice', 0):.2f}", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Stop Loss:", className="font-weight-bold mb-0"),
                                                html.P(f"${option_data.get('stopLossPrice', 0):.2f}", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Optimal Hold:", className="font-weight-bold mb-0"),
                                                html.P(f"{option_data.get('optimalHoldDays', 0)} days", className="mb-0")
                                            ],
                                            className="col-md-4"
                                        )
                                    ],
                                    className="row mb-3"
                                )
                            ],
                            className="mb-4"
                        ),
                        
                        # Signal details section
                        html.Div(
                            [
                                html.H5("Signal Details", className="border-bottom pb-2"),
                                html.Ul(
                                    [html.Li(detail) for detail in signal_details] if signal_details else [html.Li("No signal details available.")],
                                    className="pl-3"
                                )
                            ],
                            className="mb-4"
                        )
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id=f"close-modal-{option_data.get('symbol', '')}-{option_data.get('strikePrice', 0)}", className="ml-auto")
                )
            ],
            id=f"modal-{option_data.get('symbol', '')}-{option_data.get('strikePrice', 0)}",
            size="lg"
        )
        
        return modal
    
    def create_recommendations_grid(self, recommendations, recommendation_type='all'):
        """
        Create a grid of recommendation cards
        
        Args:
            recommendations (dict): Recommendations data
            recommendation_type (str): Type of recommendations to display ('all', 'calls', 'puts')
            
        Returns:
            dash component: Grid of recommendation cards
        """
        if not recommendations:
            return html.Div(
                html.P("No recommendations available. Please enter a symbol and select timeframe."),
                className="text-center my-5"
            )
        
        symbol = recommendations.get('symbol', '')
        calls = recommendations.get('calls', [])
        puts = recommendations.get('puts', [])
        
        # Filter based on recommendation type
        if recommendation_type == 'calls':
            puts = []
        elif recommendation_type == 'puts':
            calls = []
        
        # Create cards
        call_cards = [self.create_recommendation_card(call, 'CALL') for call in calls]
        put_cards = [self.create_recommendation_card(put, 'PUT') for put in puts]
        
        # Create modals for details
        call_modals = [self.create_recommendation_detail_modal(call, 'CALL') for call in calls]
        put_modals = [self.create_recommendation_detail_modal(put, 'PUT') for put in puts]
        
        # Combine all cards
        all_cards = call_cards + put_cards
        all_modals = call_modals + put_modals
        
        if not all_cards:
            return html.Div(
                html.P(f"No {recommendation_type} recommendations available for {symbol}."),
                className="text-center my-5"
            )
        
        # Create grid layout
        grid = html.Div(
            [
                html.Div(
                    [
                        html.H3(f"Top Recommendations for {symbol}", className="mb-4"),
                        html.Div(
                            [
                                dbc.Button("All", id="btn-all", color="primary", className="mr-2", 
                                          active=recommendation_type == 'all'),
                                dbc.Button("Calls", id="btn-calls", color="success", className="mr-2", 
                                          active=recommendation_type == 'calls'),
                                dbc.Button("Puts", id="btn-puts", color="danger", 
                                          active=recommendation_type == 'puts')
                            ],
                            className="mb-4"
                        )
                    ],
                    className="d-flex justify-content-between align-items-center flex-wrap"
                ),
                html.Div(
                    [
                        html.Div(
                            card,
                            className="col-md-4"
                        ) for card in all_cards
                    ],
                    className="row"
                ),
                html.Div(all_modals)
            ]
        )
        
        return grid
    
    def create_profit_projection_chart(self, option_data):
        """
        Create a profit projection chart for an option
        
        Args:
            option_data (dict): Option data
            
        Returns:
            dash component: Profit projection chart
        """
        # Extract data
        entry_price = option_data.get('entryPrice', 0)
        days_to_expiration = option_data.get('daysToExpiration', 30)
        option_type = option_data.get('optionType', 'CALL').upper()
        
        # Create time points (days)
        days = list(range(0, min(days_to_expiration + 1, 60), 1))
        
        # Simulate price decay (simplified model)
        if entry_price > 0 and days_to_expiration > 0:
            # Simplified theta decay model (accelerates closer to expiration)
            prices = []
            for day in days:
                remaining_days = days_to_expiration - day
                if remaining_days <= 0:
                    prices.append(0)  # At expiration, time value is zero
                else:
                    # Square root model for time decay
                    time_factor = (remaining_days / days_to_expiration) ** 0.5
                    # Assume 70% of option price is time value (simplified)
                    intrinsic_value = entry_price * 0.3
                    time_value = entry_price * 0.7 * time_factor
                    prices.append(intrinsic_value + time_value)
        else:
            # Fallback if no valid data
            prices = [entry_price] * len(days)
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=days,
                y=prices,
                mode='lines',
                name='Option Price',
                line=dict(color='#007bff', width=3)
            )
        )
        
        # Add entry price line
        fig.add_trace(
            go.Scatter(
                x=[0, days[-1]],
                y=[entry_price, entry_price],
                mode='lines',
                name='Entry Price',
                line=dict(color='#6c757d', width=2, dash='dash')
            )
        )
        
        # Add take profit and stop loss if available
        take_profit = option_data.get('takeProfitPrice', 0)
        stop_loss = option_data.get('stopLossPrice', 0)
        
        if take_profit > 0:
            fig.add_trace(
                go.Scatter(
                    x=[0, days[-1]],
                    y=[take_profit, take_profit],
                    mode='lines',
                    name='Take Profit',
                    line=dict(color='#28a745', width=2, dash='dash')
                )
            )
        
        if stop_loss > 0:
            fig.add_trace(
                go.Scatter(
                    x=[0, days[-1]],
                    y=[stop_loss, stop_loss],
                    mode='lines',
                    name='Stop Loss',
                    line=dict(color='#dc3545', width=2, dash='dash')
                )
            )
        
        # Add optimal hold day marker if available
        optimal_hold = option_data.get('optimalHoldDays', 0)
        if optimal_hold > 0 and optimal_hold < len(days):
            optimal_price = prices[optimal_hold]
            fig.add_trace(
                go.Scatter(
                    x=[optimal_hold],
                    y=[optimal_price],
                    mode='markers',
                    name='Optimal Exit',
                    marker=dict(color='#17a2b8', size=12, symbol='star')
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{option_type} ${option_data.get('strikePrice', 0):.2f} Price Projection",
            xaxis_title="Days from Now",
            yaxis_title="Option Price ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )
        
        # Create chart component
        chart = dcc.Graph(
            figure=fig,
            config={'displayModeBar': False}
        )
        
        return chart
