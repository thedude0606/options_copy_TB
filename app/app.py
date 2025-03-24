"""
Main application layout for options recommendation platform.
Integrates all components and tabs into a cohesive dashboard.
"""
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from app.components.recommendations_tab import create_recommendations_tab
from app.components.trade_card import create_trade_cards_container

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Define app layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Options Recommendation Platform", className="dashboard-title"),
            html.P("Powered by Schwab API", className="dashboard-subtitle")
        ], width=12, className="header-container")
    ], className="header-row mb-4"),
    
    # Main content with tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                # Recommendations Tab
                dbc.Tab(
                    create_recommendations_tab(),
                    label="Recommendations",
                    tab_id="tab-recommendations"
                ),
                
                # Real-Time Data Tab (from existing code)
                dbc.Tab(
                    html.Div(id="real-time-tab-content"),
                    label="Real-Time Data",
                    tab_id="tab-real-time"
                ),
                
                # Technical Indicators Tab
                dbc.Tab(
                    html.Div(id="indicators-tab-content"),
                    label="Technical Indicators",
                    tab_id="tab-indicators"
                ),
                
                # Greeks Analysis Tab
                dbc.Tab(
                    html.Div(id="greeks-tab-content"),
                    label="Greeks Analysis",
                    tab_id="tab-greeks"
                ),
                
                # Historical Data Tab (from existing code)
                dbc.Tab(
                    html.Div(id="historical-tab-content"),
                    label="Historical Data",
                    tab_id="tab-historical"
                )
            ], id="main-tabs", active_tab="tab-recommendations")
        ], width=12)
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Options Recommendation Platform © 2025", className="text-center text-muted")
        ], width=12)
    ], className="mt-4"),
    
    # Store components for holding data
    dcc.Store(id="options-data"),
    dcc.Store(id="historical-data"),
    dcc.Store(id="quote-data"),
    dcc.Store(id="recommendations-data")
], fluid=True, className="dashboard-container")

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Options Recommendation Platform</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dashboard styles */
            .dashboard-container {
                padding: 20px;
            }
            .dashboard-title {
                color: #2c3e50;
                font-weight: 700;
            }
            .dashboard-subtitle {
                color: #7f8c8d;
                font-size: 1.2rem;
            }
            
            /* Trade card styles */
            .trade-card {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .trade-card:hover {
                transform: translateY(-5px);
            }
            .trade-card .section {
                margin-bottom: 15px;
            }
            .trade-card .section-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 8px;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }
            .trade-card .label {
                font-weight: 600;
                color: #555;
            }
            .trade-card .value {
                font-weight: 400;
                color: #333;
            }
            
            /* Filter container styles */
            .filter-container {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #e9ecef;
            }
            
            /* Status message styles */
            .status-message {
                padding: 10px;
                border-radius: 5px;
            }
            .text-success {
                background-color: #d4edda;
                color: #155724;
            }
            .text-warning {
                background-color: #fff3cd;
                color: #856404;
            }
            .text-danger {
                background-color: #f8d7da;
                color: #721c24;
            }
            
            /* Sidebar styles */
            .sidebar {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
            }
            
            /* Recommendations container */
            .recommendations-container {
                min-height: 400px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
