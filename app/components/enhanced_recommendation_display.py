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
        
        # Get exit strategy data if available
        exit_strategy = option_data.get('exitStrategy', {})
        has_exit_strategy = len(exit_strategy) > 0
        
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
        body_content = [
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
            )
        ]
        
        # Add exit strategy section if available
        if has_exit_strategy:
            # Format exit date
            exit_date = exit_strategy.get('optimalExitDate', '')
            if exit_date:
                try:
                    if isinstance(exit_date, str):
                        exit_date_obj = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
                        formatted_exit_date = exit_date_obj.strftime('%b %d, %Y')
                    else:
                        formatted_exit_date = exit_date.strftime('%b %d, %Y')
                except:
                    formatted_exit_date = str(exit_date)
            else:
                formatted_exit_date = "N/A"
            
            # Get days to hold
            days_to_hold = exit_strategy.get('daysToHold', 0)
            
            # Add exit strategy section
            exit_strategy_section = html.Div(
                [
                    html.Hr(className="my-2"),
                    html.P("Exit Strategy", className="mb-1 font-weight-bold"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("Exit Date", className="mb-0 small"),
                                    html.P(formatted_exit_date, className="mb-0 font-weight-bold")
                                ],
                                className="col-6"
                            ),
                            html.Div(
                                [
                                    html.P("Days to Hold", className="mb-0 small"),
                                    html.P(f"{days_to_hold}", className="mb-0 font-weight-bold")
                                ],
                                className="col-6"
                            )
                        ],
                        className="row mb-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("Take Profit", className="mb-0 small"),
                                    html.P(f"${exit_strategy.get('takeProfit', 0):.2f}", 
                                          className="mb-0 font-weight-bold text-success")
                                ],
                                className="col-6"
                            ),
                            html.Div(
                                [
                                    html.P("Stop Loss", className="mb-0 small"),
                                    html.P(f"${exit_strategy.get('stopLoss', 0):.2f}", 
                                          className="mb-0 font-weight-bold text-danger")
                                ],
                                className="col-6"
                            )
                        ],
                        className="row mb-2"
                    )
                ],
                className="mb-3"
            )
            
            body_content.append(exit_strategy_section)
        
        # Add action buttons
        body_content.append(
            html.Div(
                [
                    dbc.Button("View Details", color="light", className="mr-2", id=f"view-details-{option_data.get('symbol', '')}-{option_data.get('strikePrice', 0)}"),
                    dbc.Button("Trade Now", color="primary", id=f"trade-now-{option_data.get('symbol', '')}-{option_data.get('strikePrice', 0)}")
                ],
                className="d-flex justify-content-between"
            )
        )
        
        # Create card body
        body = dbc.CardBody(body_content)
        
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
        
        # Get exit strategy data if available
        exit_strategy = option_data.get('exitStrategy', {})
        has_exit_strategy = len(exit_strategy) > 0
        
        # Create modal body content
        modal_body_content = [
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
                                    html.P("Delta:", className="font-weight-bold mb-0"),
                                    html.P(f"{option_data.get('delta', 0):.3f}", className="mb-0")
                                ],
                                className="col-md-4"
                            ),
                            html.Div(
                                [
                                    html.P("Implied Vol:", className="font-weight-bold mb-0"),
                                    html.P(f"{option_data.get('impliedVolatility', 0) * 100:.1f}%", className="mb-0")
                                ],
                                className="col-md-4"
                            )
                        ],
                        className="row mb-3"
                    )
                ],
                className="mb-4"
            ),
            
            # Recommendation details section
            html.Div(
                [
                    html.H5("Recommendation Details", className="border-bottom pb-2"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("Confidence:", className="font-weight-bold mb-0"),
                                    html.P(f"{option_data.get('confidenceScore', 0):.0f}%", className="mb-0")
                                ],
                                className="col-md-4"
                            ),
                            html.Div(
                                [
                                    html.P("Expected Return:", className="font-weight-bold mb-0"),
                                    html.P(f"{option_data.get('expectedReturn', 0):.1f}%", 
                                          className="mb-0",
                                          style={"color": "#28a745" if option_data.get('expectedReturn', 0) > 0 else "#dc3545"})
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
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("Risk/Reward:", className="font-weight-bold mb-0"),
                                    html.P(f"{option_data.get('riskRewardRatio', 0):.2f}x", className="mb-0")
                                ],
                                className="col-md-4"
                            ),
                            html.Div(
                                [
                                    html.P("Recommendation:", className="font-weight-bold mb-0"),
                                    html.P(f"{option_data.get('recommendationType', 'BUY')}", 
                                          className="mb-0 font-weight-bold",
                                          style={"color": "#28a745" if option_data.get('recommendationType', '') == 'BUY' else "#dc3545"})
                                ],
                                className="col-md-4"
                            ),
                            html.Div(
                                [
                                    html.P("Strategy:", className="font-weight-bold mb-0"),
                                    html.P(f"{option_data.get('strategyType', 'Momentum')}", className="mb-0")
                                ],
                                className="col-md-4"
                            )
                        ],
                        className="row mb-3"
                    )
                ],
                className="mb-4"
            )
        ]
        
        # Add exit strategy section if available
        if has_exit_strategy:
            # Format exit date
            exit_date = exit_strategy.get('optimalExitDate', '')
            if exit_date:
                try:
                    if isinstance(exit_date, str):
                        exit_date_obj = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
                        formatted_exit_date = exit_date_obj.strftime('%b %d, %Y')
                    else:
                        formatted_exit_date = exit_date.strftime('%b %d, %Y')
                except:
                    formatted_exit_date = str(exit_date)
            else:
                formatted_exit_date = "N/A"
            
            # Get price targets
            price_targets = exit_strategy.get('priceTargets', [])
            
            # Create exit strategy section
            exit_strategy_section = html.Div(
                [
                    html.H5("Exit Strategy", className="border-bottom pb-2"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("Optimal Exit Date:", className="font-weight-bold mb-0"),
                                    html.P(formatted_exit_date, className="mb-0")
                                ],
                                className="col-md-4"
                            ),
                            html.Div(
                                [
                                    html.P("Days to Hold:", className="font-weight-bold mb-0"),
                                    html.P(f"{exit_strategy.get('daysToHold', 0)}", className="mb-0")
                                ],
                                className="col-md-4"
                            ),
                            html.Div(
                                [
                                    html.P("Exit Probability:", className="font-weight-bold mb-0"),
                                    html.P(f"{exit_strategy.get('exitProbability', 0) * 100:.0f}%", className="mb-0")
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
                                    html.P("Take Profit:", className="font-weight-bold mb-0"),
                                    html.P(f"${exit_strategy.get('takeProfit', 0):.2f}", 
                                          className="mb-0 text-success")
                                ],
                                className="col-md-4"
                            ),
                            html.Div(
                                [
                                    html.P("Stop Loss:", className="font-weight-bold mb-0"),
                                    html.P(f"${exit_strategy.get('stopLoss', 0):.2f}", 
                                          className="mb-0 text-danger")
                                ],
                                className="col-md-4"
                            ),
                            html.Div(
                                [
                                    html.P("Confidence:", className="font-weight-bold mb-0"),
                                    html.P(f"{exit_strategy.get('confidenceScore', 0) * 100:.0f}%", className="mb-0")
                                ],
                                className="col-md-4"
                            )
                        ],
                        className="row mb-3"
                    )
                ],
                className="mb-4"
            )
            
            # Add price targets table if available
            if price_targets:
                price_targets_rows = []
                for i, target in enumerate(price_targets):
                    price_targets_rows.append(
                        html.Tr([
                            html.Td(f"Target {i+1}"),
                            html.Td(f"${target.get('price', 0):.2f}"),
                            html.Td(f"{target.get('profit_percentage', 0):.1f}%"),
                            html.Td(f"{target.get('percentage', 0) * 100:.0f}%")
                        ])
                    )
                
                price_targets_table = html.Div(
                    [
                        html.H6("Price Targets", className="mt-3 mb-2"),
                        html.Table(
                            [
                                html.Thead(
                                    html.Tr([
                                        html.Th("Target"),
                                        html.Th("Price"),
                                        html.Th("Profit %"),
                                        html.Th("Position %")
                                    ])
                                ),
                                html.Tbody(price_targets_rows)
                            ],
                            className="table table-sm"
                        )
                    ]
                )
                
                exit_strategy_section.children.append(price_targets_table)
            
            # Add profit projections if available
            profit_projections = exit_strategy.get('profit_projections', {})
            if profit_projections:
                scenarios = profit_projections.get('scenarios', [])
                if scenarios:
                    scenario_rows = []
                    for scenario in scenarios:
                        scenario_rows.append(
                            html.Tr([
                                html.Td(scenario.get('description', '')),
                                html.Td(f"${scenario.get('exit_price', 0):.2f}"),
                                html.Td(f"{scenario.get('profit_percentage', 0):.1f}%", 
                                       style={"color": "#28a745" if scenario.get('profit_percentage', 0) > 0 else "#dc3545"}),
                                html.Td(f"{scenario.get('probability', 0) * 100:.0f}%"),
                                html.Td(f"{scenario.get('days_to_target', 0):.1f}")
                            ])
                        )
                    
                    profit_projections_table = html.Div(
                        [
                            html.H6("Profit Projections", className="mt-3 mb-2"),
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr([
                                            html.Th("Scenario"),
                                            html.Th("Exit Price"),
                                            html.Th("Profit %"),
                                            html.Th("Probability"),
                                            html.Th("Days to Target")
                                        ])
                                    ),
                                    html.Tbody(scenario_rows)
                                ],
                                className="table table-sm"
                            ),
                            html.Div(
                                [
                                    html.P(f"Expected Value: ${profit_projections.get('expected_value', 0):.2f}", 
                                          className="mb-0 font-weight-bold"),
                                    html.P(f"Risk-Reward Ratio: {profit_projections.get('risk_reward_ratio', 0):.2f}x", 
                                          className="mb-0")
                                ],
                                className="mt-2"
                            )
                        ]
                    )
                    
                    exit_strategy_section.children.append(profit_projections_table)
            
            # Add exit reasons if available
            exit_reasons = exit_strategy.get('exitReasons', [])
            if exit_reasons:
                exit_reasons_list = html.Div(
                    [
                        html.H6("Exit Reasons", className="mt-3 mb-2"),
                        html.Ul([html.Li(reason) for reason in exit_reasons], className="pl-3")
                    ]
                )
                
                exit_strategy_section.children.append(exit_reasons_list)
            
            # Add exit strategy section to modal body
            modal_body_content.append(exit_strategy_section)
        
        # Add signal details section if available
        if signal_details:
            signal_details_section = html.Div(
                [
                    html.H5("Signal Details", className="border-bottom pb-2"),
                    html.Ul([html.Li(detail) for detail in signal_details], className="pl-3")
                ],
                className="mb-4"
            )
            
            modal_body_content.append(signal_details_section)
        
        # Create modal
        modal = dbc.Modal(
            [
                dbc.ModalHeader(
                    f"{option_data.get('symbol', '')} {option_type} ${option_data.get('strikePrice', 0):.2f}",
                    style={"background-color": color, "color": "white"}
                ),
                dbc.ModalBody(modal_body_content),
                dbc.ModalFooter(
                    dbc.Button("Close", id=f"close-modal-{option_data.get('symbol', '')}", className="ml-auto")
                )
            ],
            id=f"modal-{option_data.get('symbol', '')}-{option_data.get('strikePrice', 0)}",
            size="lg"
        )
        
        return modal
    
    def create_exit_strategy_visualization(self, option_data, option_type='CALL'):
        """
        Create a visualization for the exit strategy
        
        Args:
            option_data (dict): Option data
            option_type (str): Option type (CALL or PUT)
            
        Returns:
            dash component: Graph with exit strategy visualization
        """
        # Get exit strategy data
        exit_strategy = option_data.get('exitStrategy', {})
        if not exit_strategy:
            return html.Div("No exit strategy data available")
        
        # Get price targets
        price_targets = exit_strategy.get('priceTargets', [])
        
        # Get entry price and dates
        entry_price = option_data.get('entryPrice', 0)
        entry_date_str = exit_strategy.get('entry_date', '')
        exit_date_str = exit_strategy.get('optimal_exit_time', '')
        
        # Parse dates
        try:
            if isinstance(entry_date_str, str):
                entry_date = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00'))
            else:
                entry_date = datetime.now()
                
            if isinstance(exit_date_str, str):
                exit_date = datetime.fromisoformat(exit_date_str.replace('Z', '+00:00'))
            else:
                exit_date = entry_date + timedelta(days=exit_strategy.get('days_to_hold', 7))
        except:
            entry_date = datetime.now()
            exit_date = entry_date + timedelta(days=exit_strategy.get('days_to_hold', 7))
        
        # Generate date range
        date_range = pd.date_range(start=entry_date, end=exit_date, freq='D')
        
        # Get stop loss and take profit
        stop_loss = exit_strategy.get('stopLoss', entry_price * 0.8)
        take_profit = exit_strategy.get('takeProfit', entry_price * 1.2)
        
        # Generate price paths
        # Expected path (linear from entry to take profit)
        expected_path = np.linspace(entry_price, take_profit, len(date_range))
        
        # Optimistic path (faster to take profit)
        optimistic_path = np.array([entry_price + (take_profit - entry_price) * (i/len(date_range))**0.5 for i in range(len(date_range))])
        
        # Pessimistic path (towards stop loss then recovery)
        mid_point = len(date_range) // 3
        pessimistic_path = np.array([
            entry_price - (entry_price - stop_loss) * (i/mid_point) if i < mid_point else
            stop_loss + (take_profit - stop_loss) * ((i - mid_point)/(len(date_range) - mid_point))
            for i in range(len(date_range))
        ])
        
        # Create figure
        fig = go.Figure()
        
        # Add price paths
        fig.add_trace(go.Scatter(
            x=date_range, 
            y=expected_path,
            mode='lines',
            name='Expected Path',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=date_range, 
            y=optimistic_path,
            mode='lines',
            name='Optimistic Path',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=date_range, 
            y=pessimistic_path,
            mode='lines',
            name='Pessimistic Path',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add entry point
        fig.add_trace(go.Scatter(
            x=[entry_date],
            y=[entry_price],
            mode='markers',
            name='Entry Point',
            marker=dict(color='blue', size=10)
        ))
        
        # Add exit point
        fig.add_trace(go.Scatter(
            x=[exit_date],
            y=[take_profit],
            mode='markers',
            name='Target Exit',
            marker=dict(color='green', size=10)
        ))
        
        # Add stop loss line
        fig.add_shape(
            type="line",
            x0=entry_date,
            y0=stop_loss,
            x1=exit_date,
            y1=stop_loss,
            line=dict(color="red", width=2, dash="dot")
        )
        
        # Add take profit line
        fig.add_shape(
            type="line",
            x0=entry_date,
            y0=take_profit,
            x1=exit_date,
            y1=take_profit,
            line=dict(color="green", width=2, dash="dot")
        )
        
        # Add price targets
        for i, target in enumerate(price_targets):
            target_price = target.get('price', 0)
            target_date = entry_date + timedelta(days=exit_strategy.get('days_to_hold', 7) * (i+1) / (len(price_targets)+1))
            
            fig.add_trace(go.Scatter(
                x=[target_date],
                y=[target_price],
                mode='markers',
                name=f'Target {i+1}',
                marker=dict(color='purple', size=8, symbol='triangle-up')
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Exit Strategy for {option_data.get('symbol', '')} {option_type}",
            xaxis_title="Date",
            yaxis_title="Option Price ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="closest"
        )
        
        # Create graph component
        graph = dcc.Graph(
            figure=fig,
            config={'displayModeBar': False}
        )
        
        return graph
