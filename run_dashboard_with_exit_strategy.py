import os
import sys
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from app.components.exit_strategy_display import create_exit_strategy_tab, register_exit_strategy_callbacks, ExitStrategyDisplay

# Import recommendation engine
from app.analysis.exit_strategy_recommendation_engine import ExitStrategyEnhancedRecommendationEngine

# Import data collector
from app.data.options_collector import OptionsDataCollector

def main():
    # Initialize data collector
    data_collector = OptionsDataCollector()
    
    # Initialize recommendation engine with exit strategy prediction
    recommendation_engine = ExitStrategyEnhancedRecommendationEngine(
        data_collector=data_collector,
        ml_config_path='config/ml_config.json',
        debug=True
    )
    
    # Initialize Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Define app layout
    app.layout = html.Div([
        html.H1("Options Trading Platform with Exit Strategy Prediction", className="mb-4"),
        
        dcc.Tabs([
            dcc.Tab(label="Recommendations", children=[
                html.Div([
                    html.H3("Options Recommendations", className="mb-4"),
                    html.Div(id="recommendations-output")
                ])
            ]),
            dcc.Tab(label="Exit Strategies", children=create_exit_strategy_tab()),
            # Other tabs can be added here
        ]),
        
        # Interval component for periodic updates
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # in milliseconds (1 minute)
            n_intervals=0
        )
    ])
    
    # Register callbacks
    register_exit_strategy_callbacks(app, recommendation_engine)
    
    # Callback for recommendations tab
    @app.callback(
        Output("recommendations-output", "children"),
        [Input("interval-component", "n_intervals")]
    )
    def update_recommendations(n_intervals):
        # Generate recommendations with exit strategies
        try:
            recommendations = recommendation_engine.generate_recommendations(
                symbol='AAPL',  # Example symbol
                lookback_days=30,
                confidence_threshold=0.6,
                max_recommendations=5
            )
            
            # If no recommendations, return message
            if recommendations.empty:
                return html.Div("No recommendations available at this time.")
            
            # Convert to list of dictionaries
            rec_list = recommendations.to_dict('records')
            
            # Create cards for each recommendation
            cards = []
            for rec in rec_list:
                # Create recommendation card
                rec_card = dbc.Card(
                    dbc.CardBody([
                        html.H5(f"{rec.get('underlying', '')} {rec.get('option_type', '')} {rec.get('strike', 0)}", className="card-title"),
                        html.H6(f"Expiration: {rec.get('expiration_date', '')}", className="card-subtitle mb-2 text-muted"),
                        
                        html.Div([
                            html.Strong("Entry Price:"),
                            html.Span(f" ${rec.get('price', 0):.2f}", className="ms-2")
                        ], className="mb-2"),
                        
                        html.Div([
                            html.Strong("Confidence:"),
                            html.Span(f" {rec.get('confidence', 0) * 100:.1f}%", className="ms-2")
                        ], className="mb-2"),
                        
                        html.Hr(),
                        
                        # Add exit strategy display
                        ExitStrategyDisplay.create_exit_strategy_card(rec)
                    ])
                )
                cards.append(dbc.Col(rec_card, width=6, className="mb-4"))
            
            # Arrange cards in rows
            rows = []
            for i in range(0, len(cards), 2):
                row_cards = cards[i:i+2]
                rows.append(dbc.Row(row_cards, className="mb-4"))
            
            return html.Div(rows)
            
        except Exception as e:
            return html.Div(f"Error generating recommendations: {str(e)}")
    
    # Run the app
    app.run_server(debug=True, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()
