# Design Decisions

## Architecture Choices

- **Repository Structure**: Created a new repository based on the Schwabdev library to maintain separation of concerns while leveraging the existing API wrapper functionality.

- **Documentation Approach**: Implementing three key documentation files (PROGRESS.md, TODO.md, DECISIONS.md) to track development progress, pending tasks, and architectural decisions.

- **Authentication System**: Implemented a robust authentication system that handles OAuth flow with both interactive and non-interactive modes, with token persistence for improved user experience.

- **Simplified Application Structure**: Moved from a complex package-based structure to a standalone script approach to resolve import issues and ensure compatibility across different environments.

## Technology Selections

- **API Integration**: Using the Schwabdev library as the foundation for Schwab API integration to leverage existing authentication and request handling.

- **Dashboard Framework**: Using Dash with Plotly for creating the interactive dashboard due to its:
  - Python integration capabilities
  - Interactive visualization components
  - Real-time update functionality
  - Ease of deployment on localhost

- **Data Storage**: Using in-memory data structures for real-time data with token persistence for authentication.

- **Dependency Management**: Updated from strict version requirements to more flexible version specifications to improve compatibility across different Python environments and versions.

## Design Patterns

- **Environment Variables**: Using .env file for secure credential management to avoid hardcoding sensitive information.

- **Modular Design**: Organized code into logical classes (SchwabAuth, OptionsDataRetriever) for better maintainability even within a single file.

- **Event-Driven Updates**: Implemented event-driven pattern with Dash callbacks for real-time data updates in the dashboard.

- **Singleton Pattern**: Used a singleton instance for the authentication manager to ensure consistent authentication state.

- **Self-Contained Application**: Consolidated all functionality into a single file to eliminate import issues while maintaining clean code organization.

## Rationale for Key Decisions

- **Local Development**: Developing on localhost:8050 for ease of testing and development before any potential deployment.

- **API Authentication Approach**: Implemented a user-friendly OAuth flow that provides clear instructions and handles token persistence to minimize authentication friction.

- **Data Visualization Strategy**: Created a tabbed interface to organize different types of options data (chain, Greeks, historical) for better user experience and clarity.

- **Error Handling**: Implemented comprehensive error handling in data retrieval functions to ensure the dashboard remains functional even if API calls fail.

- **Flexible Dependencies**: Changed from strict version requirements to more flexible version specifications to prevent installation issues across different environments and Python versions.

- **Standalone Script Approach**: After encountering persistent import issues with the package-based structure, switched to a self-contained script approach that eliminates import complexities while maintaining all functionality.

- **API Method Naming Conventions**: Updated API method calls to match the Schwabdev library's actual method names:
  - Changed `client.get_option_chain` to `client.option_chains` 
  - Changed `client.get_price_history` to `client.price_history`
  - Changed `client.get_quote` to `client.quote`
  
  This decision was based on examining the Schwabdev documentation and examples which showed that the library uses shorter method names without the "get_" prefix. This ensures compatibility with the latest version of the Schwabdev library and follows its established naming conventions.

- **Parameter Name Correction**: Updated the parameter name in the option_chains() method call:
  - Changed `includeQuotes` to `includeUnderlyingQuote`
  
  This decision was based on examining the Schwabdev library's client.py file, which revealed that the correct parameter name is 'includeUnderlyingQuote' rather than 'includeQuotes'. This ensures compatibility with the Schwabdev library's API and resolves the error: "Client.option_chains() got an unexpected keyword argument 'includeQuotes'".

- **Historical Data Parameter Addition**: Added the required 'periodType' parameter to the price_history() method call:
  - Added `periodType="day"` to the price_history() method parameters
  
  This decision was based on examining the Schwabdev library's client.py file, which revealed that the price_history() method requires a 'periodType' parameter that was missing in our implementation. Adding this parameter ensures compatibility with the Schwabdev library's API and enables the historical data tab to properly display price history data.
