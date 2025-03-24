"""
Trade card component for options recommendation platform.
Implements the UI for displaying option recommendations in a card format.
"""
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import json
import ast
from datetime import datetime

def create_trade_card(recommendation):
    """
    Create a trade card component for an option recommendation
    
    Args:
        recommendation (dict): Option recommendation data
        
    Returns:
        dbc.Card: Trade card component
    """
    try:
        # Parse recommendation details if it's a string
        if isinstance(recommendation.get('recommendation_details', ''), str):
            try:
                details = ast.literal_eval(recommendation['recommendation_details'])
            except:
                details = {}
        else:
            details = recommendation.get('recommendation_details', {})
        
        # Extract data
        option_type = details.get('type', 'UNKNOWN').upper()
        strike = details.get('strike', 0)
        expiration = details.get('expiration', 'UNKNOWN')
        current_price = details.get('current_price', 0)
        underlying_price = details.get('underlying_price', 0)
        probability = details.get('probability_of_profit', 0)
        risk_reward = details.get('risk_reward_ratio', 0)
        potential_return = details.get('potential_return_pct', 0)
        confidence = details.get('confidence_score', 0)
        signal_details = details.get('signal_details', {})
        
        # Format expiration date
        try:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            formatted_exp = exp_date.strftime('%b %d, %Y')
        except:
            formatted_exp = expiration
        
        # Determine card color based on option type
        if option_type == 'CALL':
            card_color = 'success'  # Green for calls
            header_color = '#28a745'
        elif option_type == 'PUT':
            card_color = 'danger'  # Red for puts
            header_color = '#dc3545'
        else:
            card_color = 'info'  # Blue for unknown
            header_color = '#17a2b8'
        
        # Create card header
        header = dbc.CardHeader(
            [
                html.H4(f"{option_type} - Strike ${strike:.2f}", className="card-title"),
                html.H6(f"Expires: {formatted_exp}", className="card-subtitle")
            ],
            style={"background-color": header_color, "color": "white"}
        )
        
        # Create card body
        body = dbc.CardBody(
            [
                # Price information
                html.Div([
                    html.H5("Price Information", className="section-title"),
                    html.P([
                        html.Span("Entry Price: ", className="label"),
                        html.Span(f"${current_price:.2f}", className="value")
                    ]),
                    html.P([
                        html.Span("Underlying Price: ", className="label"),
                        html.Span(f"${underlying_price:.2f}", className="value")
                    ])
                ], className="section"),
                
                # Probability and risk metrics
                html.Div([
                    html.H5("Risk Metrics", className="section-title"),
                    html.P([
                        html.Span("Probability of Success: ", className="label"),
                        html.Span(f"{probability*100:.1f}%", className="value")
                    ]),
                    html.P([
                        html.Span("Risk-Reward Ratio: ", className="label"),
                        html.Span(f"{risk_reward:.2f}", className="value")
                    ]),
                    html.P([
                        html.Span("Potential Return: ", className="label"),
                        html.Span(f"{potential_return:.1f}%", className="value")
                    ]),
                    html.P([
                        html.Span("Confidence Score: ", className="label"),
                        html.Span(f"{confidence*100:.1f}%", className="value")
                    ])
                ], className="section"),
                
                # Greeks (collapsible)
                dbc.Collapse(
                    html.Div([
                        html.H5("Greeks", className="section-title"),
                        html.P([
                            html.Span("Delta: ", className="label"),
                            html.Span(f"{details.get('delta', 0):.3f}", className="value")
                        ]),
                        html.P([
                            html.Span("Gamma: ", className="label"),
                            html.Span(f"{details.get('gamma', 0):.3f}", className="value")
                        ]),
                        html.P([
                            html.Span("Theta: ", className="label"),
                            html.Span(f"{details.get('theta', 0):.3f}", className="value")
                        ]),
                        html.P([
                            html.Span("Vega: ", className="label"),
                            html.Span(f"{details.get('vega', 0):.3f}", className="value")
                        ])
                    ], className="section"),
                    id=f"collapse-{id(recommendation)}",
                    is_open=False
                ),
                
                # Toggle button for Greeks
                dbc.Button(
                    "Show Greeks",
                    id=f"toggle-{id(recommendation)}",
                    color="link",
                    className="mt-2"
                ),
                
                # Rationale
                html.Div([
                    html.H5("Rationale", className="section-title"),
                    html.Div([
                        html.P(f"{key.upper()}: {value}")
                        for key, value in signal_details.items()
                    ])
                ], className="section mt-3")
            ]
        )
        
        # Create card footer with action buttons
        footer = dbc.CardFooter(
            [
                dbc.Button("Trade Now", color=card_color, className="mr-2"),
                dbc.Button("Add to Watchlist", color="secondary", outline=True)
            ]
        )
        
        # Combine all components into a card
        card = dbc.Card(
            [header, body, footer],
            className="mb-4 trade-card",
            style={"max-width": "400px"}
        )
        
        # Add callback for toggling Greeks
        @callback(
            Output(f"collapse-{id(recommendation)}", "is_open"),
            Output(f"toggle-{id(recommendation)}", "children"),
            Input(f"toggle-{id(recommendation)}", "n_clicks"),
            State(f"collapse-{id(recommendation)}", "is_open"),
            prevent_initial_call=True
        )
        def toggle_collapse(n_clicks, is_open):
            if n_clicks:
                return not is_open, "Hide Greeks" if not is_open else "Show Greeks"
            return is_open, "Show Greeks" if not is_open else "Hide Greeks"
        
        return card
    
    except Exception as e:
        print(f"Error creating trade card: {str(e)}")
        # Return a simple error card
        return dbc.Card(
            dbc.CardBody([
                html.H5("Error Creating Trade Card", className="card-title"),
                html.P(f"Error: {str(e)}")
            ]),
            className="mb-4 trade-card bg-warning",
            style={"max-width": "400px"}
        )

def create_trade_cards_container(recommendations):
    """
    Create a container with multiple trade cards
    
    Args:
        recommendations (list): List of option recommendations
        
    Returns:
        html.Div: Container with trade cards
    """
    if not recommendations:
        return html.Div(
            html.P("No recommendations available. Try adjusting your criteria."),
            className="text-center my-5"
        )
    
    cards = [create_trade_card(rec) for rec in recommendations]
    
    return html.Div(
        dbc.Row(
            [dbc.Col(card, md=4) for card in cards],
            className="trade-cards-row"
        ),
        className="trade-cards-container"
    )
