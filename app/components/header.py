"""
Header component for the options recommendation platform.
Provides the top navigation bar and branding elements.
"""
from dash import html
import dash_bootstrap_components as dbc

def create_header():
    """
    Create the header component for the dashboard
    
    Returns:
        html.Div: Header component with navigation and branding
    """
    return html.Div([
        dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Img(
                            src="/assets/logo.png" if False else "",  # Placeholder for logo
                            height="40px",
                            className="mr-2"
                        ),
                        dbc.NavbarBrand("Options Recommendation Platform", className="ml-2")
                    ], width="auto"),
                ], align="center", className="g-0"),
                
                dbc.NavbarToggler(id="navbar-toggler"),
                
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
                        dbc.NavItem(dbc.NavLink("Documentation", href="#")),
                        dbc.NavItem(dbc.NavLink("Settings", href="#")),
                        dbc.NavItem(dbc.NavLink("About", href="#"))
                    ], className="ml-auto", navbar=True),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ], fluid=True),
            color="primary",
            dark=True,
            className="mb-4"
        )
    ])
