"""
Sidebar component for the options recommendation platform.
Provides navigation and filtering options for the dashboard.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

def create_sidebar():
    """
    Create the sidebar component for the dashboard
    
    Returns:
        html.Div: Sidebar component with navigation and filters
    """
    return html.Div([
        html.H5("Navigation", className="mb-3"),
        
        # Main navigation links
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-home me-2"), "Dashboard"],
                    href="#",
                    active="exact",
                    className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-chart-line me-2"), "Analysis"],
                    href="#",
                    active="exact",
                    className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-list me-2"), "Watchlist"],
                    href="#",
                    active="exact",
                    className="mb-2"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-history me-2"), "History"],
                    href="#",
                    active="exact",
                    className="mb-2"
                ),
            ],
            vertical=True,
            pills=True,
            className="mb-4"
        ),
        
        html.Hr(),
        
        # Filters section
        html.H5("Filters", className="mb-3"),
        
        # Symbol input
        html.Label("Symbol:"),
        dbc.InputGroup([
            dbc.Input(id="sidebar-symbol-input", type="text", placeholder="Enter symbol (e.g., AAPL)"),
            dbc.Button("Go", id="sidebar-symbol-button", color="primary", size="sm")
        ], className="mb-3"),
        
        # Date range selector
        html.Label("Date Range:"),
        dcc.DatePickerRange(
            id="sidebar-date-range",
            start_date_placeholder_text="Start Date",
            end_date_placeholder_text="End Date",
            className="mb-3"
        ),
        
        # Option type selector
        html.Label("Option Type:"),
        dbc.RadioItems(
            id="sidebar-option-type",
            options=[
                {"label": "All", "value": "ALL"},
                {"label": "Calls", "value": "CALL"},
                {"label": "Puts", "value": "PUT"}
            ],
            value="ALL",
            inline=True,
            className="mb-3"
        ),
        
        # Strategy selector
        html.Label("Strategy:"),
        dbc.Select(
            id="sidebar-strategy",
            options=[
                {"label": "All Strategies", "value": "ALL"},
                {"label": "Bullish", "value": "BULLISH"},
                {"label": "Bearish", "value": "BEARISH"},
                {"label": "Neutral", "value": "NEUTRAL"},
                {"label": "Volatile", "value": "VOLATILE"}
            ],
            value="ALL",
            className="mb-3"
        ),
        
        # Apply filters button
        dbc.Button(
            "Apply Filters",
            id="sidebar-apply-filters",
            color="primary",
            className="w-100 mb-3"
        ),
        
        # Reset filters button
        dbc.Button(
            "Reset Filters",
            id="sidebar-reset-filters",
            color="secondary",
            className="w-100"
        ),
        
        html.Hr(),
        
        # Settings section
        html.H5("Settings", className="mb-3"),
        
        # Theme toggle
        dbc.FormGroup([
            dbc.Label("Theme:"),
            dbc.Checklist(
                options=[{"label": "Dark Mode", "value": 1}],
                value=[],
                id="sidebar-theme-toggle",
                switch=True,
            ),
        ], className="mb-3"),
        
        # Refresh interval
        dbc.FormGroup([
            dbc.Label("Auto Refresh:"),
            dbc.Select(
                id="sidebar-refresh-interval",
                options=[
                    {"label": "Off", "value": "0"},
                    {"label": "30 seconds", "value": "30"},
                    {"label": "1 minute", "value": "60"},
                    {"label": "5 minutes", "value": "300"},
                ],
                value="60",
            ),
        ], className="mb-3"),
    ], className="p-3 bg-light")
