# Design Decisions

## Architecture Choices

- **Repository Structure**: Created a new repository based on the Schwabdev library to maintain separation of concerns while leveraging the existing API wrapper functionality.

- **Documentation Approach**: Implementing three key documentation files (PROGRESS.md, TODO.md, DECISIONS.md) to track development progress, pending tasks, and architectural decisions.

## Technology Selections

- **API Integration**: Using the Schwabdev library as the foundation for Schwab API integration to leverage existing authentication and request handling.

- **Dashboard Framework**: Will use Dash with Plotly for creating the interactive dashboard due to its:
  - Python integration capabilities
  - Interactive visualization components
  - Real-time update functionality
  - Ease of deployment on localhost

- **Data Storage**: Will primarily use in-memory data structures for real-time data with potential for temporary caching of historical data.

## Design Patterns

- **Environment Variables**: Using .env file for secure credential management to avoid hardcoding sensitive information.

- **Modular Design**: Will separate API interaction, data processing, and UI components to improve maintainability.

- **Event-Driven Updates**: Will implement event-driven pattern for real-time data updates in the dashboard.

## Rationale for Key Decisions

- **Local Development**: Developing on localhost:8050 for ease of testing and development before any potential deployment.

- **API Authentication Approach**: Will implement authentication following Schwabdev library patterns to ensure compatibility and security.

- **Data Visualization Strategy**: Will focus on clear presentation of options data with emphasis on the Greeks for options analysis.
