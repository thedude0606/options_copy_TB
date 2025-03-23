# Design Decisions

## Architecture Choices

- **Repository Structure**: Created a new repository based on the Schwabdev library to maintain separation of concerns while leveraging the existing API wrapper functionality.

- **Documentation Approach**: Implementing three key documentation files (PROGRESS.md, TODO.md, DECISIONS.md) to track development progress, pending tasks, and architectural decisions.

- **Authentication System**: Implemented a robust authentication system that handles OAuth flow with both interactive and non-interactive modes, with token persistence for improved user experience.

## Technology Selections

- **API Integration**: Using the Schwabdev library as the foundation for Schwab API integration to leverage existing authentication and request handling.

- **Dashboard Framework**: Using Dash with Plotly for creating the interactive dashboard due to its:
  - Python integration capabilities
  - Interactive visualization components
  - Real-time update functionality
  - Ease of deployment on localhost

- **Data Storage**: Using in-memory data structures for real-time data with token persistence for authentication.

## Design Patterns

- **Environment Variables**: Using .env file for secure credential management to avoid hardcoding sensitive information.

- **Modular Design**: Separated API interaction (auth.py, options_data.py) from UI components (app.py) to improve maintainability.

- **Event-Driven Updates**: Implemented event-driven pattern with Dash callbacks for real-time data updates in the dashboard.

- **Singleton Pattern**: Used a singleton instance for the authentication manager to ensure consistent authentication state.

## Rationale for Key Decisions

- **Local Development**: Developing on localhost:8050 for ease of testing and development before any potential deployment.

- **API Authentication Approach**: Implemented a user-friendly OAuth flow that provides clear instructions and handles token persistence to minimize authentication friction.

- **Data Visualization Strategy**: Created a tabbed interface to organize different types of options data (chain, Greeks, historical) for better user experience and clarity.

- **Error Handling**: Implemented comprehensive error handling in data retrieval functions to ensure the dashboard remains functional even if API calls fail.
