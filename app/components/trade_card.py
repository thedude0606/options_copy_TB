"""
Trade card component for options recommendation platform.
Implements the UI for displaying option trade recommendations.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

def create_trade_card(recommendation):
    """
    Create a trade card for an option recommendation
    
    Args:
        recommendation (dict): Option recommendation data
        
    Returns:
        html.Div: Trade card component
    """
    try:
        # Extract recommendation data
        symbol = recommendation.get('symbol', 'UNKNOWN')
        option_type = recommendation.get('optionType', 'UNKNOWN')
        strike = recommendation.get('strikePrice', 0)
        expiration = recommendation.get('expirationDate', 'Unknown')
        confidence = recommendation.get('confidence', 0)
        potential_return = recommendation.get('potentialReturn', 0)
        risk_reward = recommendation.get('riskRewardRatio', 0)
        current_price = recommendation.get('currentPrice', 0)
        
        # Extract ML-enhanced data if available
        ml_confidence = recommendation.get('mlConfidence', None)
        ml_predicted_return = recommendation.get('mlPredictedReturn', None)
        ml_risk_score = recommendation.get('mlRiskScore', None)
        
        # Extract risk management data if available
        recommended_contracts = recommendation.get('recommendedContracts', 1)
        max_risk_amount = recommendation.get('maxRiskAmount', 0)
        risk_percentage = recommendation.get('riskPercentage', 0)
        stop_loss = recommendation.get('stopLoss', 0)
        take_profit = recommendation.get('takeProfit', 0)
        
        # Extract signal details if available
        signal_details_raw = recommendation.get('signalDetails', {})
        if isinstance(signal_details_raw, str):
            try:
                signal_details = json.loads(signal_details_raw)
            except:
                signal_details = {}
        else:
            signal_details = signal_details_raw
            
            # Extract Greeks
            delta = recommendation.get('delta', 0)
            gamma = recommendation.get('gamma', 0)
            theta = recommendation.get('theta', 0)
            vega = recommendation.get('vega', 0)
        
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
            option_type_label = 'CALL'
        elif option_type == 'PUT':
            card_color = 'danger'  # Red for puts
            header_color = '#dc3545'
            option_type_label = 'PUT'
        else:
            card_color = 'info'  # Blue for unknown
            header_color = '#17a2b8'
            option_type_label = 'UNKNOWN'
        
        # Format confidence as percentage
        confidence_pct = confidence * 100 if confidence <= 1 else confidence
        
        # Format potential return as percentage
        potential_return_pct = potential_return * 100 if potential_return <= 1 else potential_return
        
        # Create card header with symbol and option type
        header = html.Div([
            html.Div(symbol, className="card-symbol"),  # Use actual symbol from recommendation data
            html.Div(option_type_label, className="card-option-type")
        ], className=f"card-header {option_type.lower()}-header")
        
        # Create card body with strike price and expiration
        body = html.Div([
            # Strike price section
            html.Div([
                html.Div(f"${strike:.2f} Strike", className="strike-price"),
                html.Div(f"Exp: {formatted_exp}", className="expiration-date")
            ], className="price-section"),
            
            # Confidence meter
            html.Div([
                html.Div("Confidence", className="metric-label"),
                html.Div(className="progress-bar-container", children=[
                    html.Div(className="progress-bar", style={"width": f"{confidence_pct:.1f}%"})
                ]),
                html.Div(f"{confidence_pct:.1f}%", className="metric-value")
            ], className="confidence-section"),
            
            # Potential return
            html.Div([
                html.Div("Potential Return", className="metric-label"),
                html.Div(f"{potential_return_pct:.1f}%", className="potential-return-value")
            ], className="return-section"),
            
            # Risk/Reward ratio
            html.Div([
                html.Div("Risk/Reward", className="metric-label"),
                html.Div(f"{risk_reward:.2f}x", className="risk-reward-value")
            ], className="risk-reward-section"),
            
            # Entry price
            html.Div([
                html.Div("Entry Price", className="metric-label"),
                html.Div(f"${current_price:.2f}", className="entry-price-value")
            ], className="entry-price-section"),
            
            # View details and trade buttons
            html.Div([
                html.Button("View Details", id={"type": "view-details-btn", "index": id(recommendation)}, className="view-details-btn"),
                html.Button("Trade Now", id={"type": "trade-now-btn", "index": id(recommendation)}, className="trade-now-btn")
            ], className="card-actions")
        ], className="card-body")
        
        # Combine header and body into a card
        card = html.Div([header, body], className=f"trade-card {option_type.lower()}-card")
        
        return card
    
    except Exception as e:
        print(f"Error creating trade card: {str(e)}")
        # Return a simple error card
        return html.Div([
            html.Div("Error", className="card-header error-header"),
            html.Div([
                html.Div(f"Error creating trade card: {str(e)}", className="error-message")
            ], className="card-body")
        ], className="trade-card error-card")

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
            className="no-recommendations-message"
        )
    
    # Create trade cards for each recommendation
    cards = [create_trade_card(rec) for rec in recommendations]
    
    # Organize cards into rows
    return html.Div(cards, className="trade-cards-grid")
