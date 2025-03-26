"""
Enhanced recommendation card component for options recommendation platform.
Provides clear call/put indicators and key trading information.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime

def create_recommendation_card(recommendation, index=0):
    """
    Create a recommendation card with clear call/put indicators
    
    Args:
        recommendation (dict): Recommendation data
        index (int): Index for callback identification
        
    Returns:
        dbc.Card: Recommendation card component
    """
    # Extract recommendation data
    symbol = recommendation.get('symbol', 'Unknown')
    option_type = recommendation.get('option_type', 'Unknown')
    strike_price = recommendation.get('strike', 0)
    expiration = recommendation.get('expiration', 'Unknown')
    confidence = recommendation.get('confidence_score', 0)
    potential_return = recommendation.get('potential_return', 0)
    risk_reward = recommendation.get('risk_reward_ratio', 0)
    entry_price = recommendation.get('entry_price', 0)
    
    # Format expiration date
    try:
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        formatted_exp = exp_date.strftime('%b %d, %Y')
    except:
        formatted_exp = expiration
    
    # Determine card color based on option type
    if option_type.upper() == 'CALL':
        header_color = 'success'
        border_color = 'success'
        option_badge_color = 'success'
    elif option_type.upper() == 'PUT':
        header_color = 'danger'
        border_color = 'danger'
        option_badge_color = 'danger'
    else:
        header_color = 'primary'
        border_color = 'primary'
        option_badge_color = 'primary'
    
    # Create confidence indicator
    confidence_percentage = int(confidence * 100)
    if confidence_percentage >= 80:
        confidence_color = 'success'
    elif confidence_percentage >= 60:
        confidence_color = 'warning'
    else:
        confidence_color = 'secondary'
    
    # Format return as percentage
    return_percentage = f"{potential_return * 100:.1f}%"
    
    return dbc.Card([
        # Card header with symbol and option type
        dbc.CardHeader([
            html.Div([
                html.H4(symbol, className="d-inline mr-2"),
                dbc.Badge(
                    option_type.upper(), 
                    color=option_badge_color,
                    className="option-type-badge"
                ),
            ], className="d-flex justify-content-between align-items-center"),
        ], className=f"bg-{header_color} text-white"),
        
        # Card body with key information
        dbc.CardBody([
            # Strike price and expiration
            html.Div([
                html.H5(f"${strike_price:.2f} Strike", className="mb-0"),
                html.Div(f"Exp: {formatted_exp}", className="text-muted small")
            ], className="mb-3"),
            
            # Key metrics
            dbc.Row([
                # Confidence score
                dbc.Col([
                    html.Div("Confidence", className="metric-label"),
                    html.Div([
                        dbc.Progress(
                            value=confidence_percentage,
                            color=confidence_color,
                            className="mb-1",
                            style={"height": "8px"}
                        ),
                        html.Span(f"{confidence_percentage}%", className="small")
                    ])
                ], width=6),
                
                # Potential return
                dbc.Col([
                    html.Div("Potential Return", className="metric-label"),
                    html.H5(
                        return_percentage,
                        className="mb-0 text-success" if potential_return > 0 else "mb-0 text-danger"
                    )
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                # Risk/Reward ratio
                dbc.Col([
                    html.Div("Risk/Reward", className="metric-label"),
                    html.Div(f"{risk_reward:.1f}x", className="metric-value")
                ], width=6),
                
                # Entry price
                dbc.Col([
                    html.Div("Entry Price", className="metric-label"),
                    html.Div(f"${entry_price:.2f}", className="metric-value")
                ], width=6)
            ], className="mb-3"),
            
            # Action buttons
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "View Details",
                        id=f"view-details-{index}",
                        color="link",
                        size="sm",
                        className="mr-2"
                    ),
                    dbc.Button(
                        "Trade Now",
                        id=f"trade-now-{index}",
                        color=header_color,
                        size="sm"
                    )
                ], className="d-flex justify-content-between")
            ])
        ]),
        
        # Store recommendation data for callbacks
        html.Div(id=f"recommendation-data-{index}", style={"display": "none"})
    ], className="recommendation-card mb-3", color=border_color, outline=True)

def create_recommendation_grid(recommendations, max_cards=6):
    """
    Create a grid of recommendation cards
    
    Args:
        recommendations (list): List of recommendation data dictionaries
        max_cards (int): Maximum number of cards to display
        
    Returns:
        html.Div: Grid of recommendation cards
    """
    if not recommendations or len(recommendations) == 0:
        return html.Div([
            html.Div(
                html.H5("No recommendations available for the current criteria", className="text-muted text-center py-5"),
                className="empty-recommendations"
            )
        ])
    
    # Limit the number of recommendations
    recommendations = recommendations[:max_cards]
    
    # Create rows of cards (3 cards per row)
    rows = []
    for i in range(0, len(recommendations), 3):
        row_cards = recommendations[i:i+3]
        row = dbc.Row([
            dbc.Col(
                create_recommendation_card(rec, index=i+j),
                width=4
            ) for j, rec in enumerate(row_cards)
        ], className="mb-3")
        rows.append(row)
    
    return html.Div(rows, className="recommendation-grid")

def create_empty_state():
    """
    Create an empty state component when no recommendations are available
    
    Returns:
        html.Div: Empty state component
    """
    return html.Div([
        html.Div([
            html.I(className="fas fa-search fa-3x mb-3 text-muted"),
            html.H4("No Recommendations Found", className="mb-2"),
            html.P("Try adjusting your filters or searching for a different symbol", className="text-muted"),
            dbc.Button("Adjust Filters", id="adjust-filters-button", color="primary", outline=True)
        ], className="text-center py-5")
    ], className="empty-state-container")
