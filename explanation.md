# Phase 2 Project Explanation

## Overview

Phase 2 is a Python-based web application built using FastAPI. It leverages the Langchain library with a Google Gemini LLM to create an agent that can answer questions based on data from a CSV file. The application provides a user interface for uploading CSV files and interacting with the data through a chat interface. It also includes authentication to protect certain pages.

## Key Components

*   **`main.py`:** This is the main application file that defines the FastAPI app, sets up routes, handles authentication, and integrates the Langchain agent.
*   **`requirements.txt`:** This file lists the Python dependencies required to run the application, including `fastapi`, `langchain_experimental`, `langchain_google_genai`, `pandas`, `python-dotenv`, and `uvicorn`.
*   **`vercel.json`:** This file configures the application for deployment on Vercel, specifying the build process and routing rules.
*   **`templates/`:** This directory contains Jinja2 templates for rendering HTML pages, including login, index, settings, rules, and default instructions pages.
*   **`static/`:** This directory contains static files such as CSS stylesheets.
*   **`data/`:** This directory stores the CSV data file used by the Langchain agent.

## How it Runs

1.  **Initialization:**
    *   The `main.py` file starts by importing necessary libraries and loading environment variables from a `.env` file (if available).
    *   It initializes a FastAPI app and configures middleware for CORS (Cross-Origin Resource Sharing).
    *   It sets up Jinja2 templates and static file serving.
    *   It defines an `async` lifespan function that runs during application startup and shutdown.
    *   During startup, the lifespan function attempts to load data from a CSV file specified by `CSV_FILE_PATH`. If the file doesn't exist, it creates a dummy CSV file.
    *   It then creates a Pandas DataFrame from the CSV data and initializes a Langchain agent using `create_pandas_dataframe_agent`. This agent is configured to use the Gemini LLM to answer questions based on the DataFrame.
2.  **Authentication:**
    *   The application implements authentication using cookies and sessions.
    *   The `load_credentials` function loads user credentials from a `credentials.json` file. If the file doesn't exist, it uses hardcoded default credentials.
    *   The `/login` route displays a login page, and the `/login` POST route handles login form submissions.
    *   The `verify_credentials` function checks if the provided username and password match the stored credentials.
    *   If the credentials are valid, a session is created using `create_session`, and a session ID is stored in a cookie.
    *   The `require_auth` dependency is used to protect certain routes, such as `/`, `/settings`, `/default-instructions`, and `/rules`.
    *   If a user is not authenticated, they are redirected to the login page.
    *   The `/logout` route handles user logout by deleting the session cookie and redirecting to the login page.
3.  **CSV Upload:**
    *   The `/upload` route allows users to upload a CSV file.
    *   The uploaded file is saved to the `data/` directory with a unique filename.
    *   The `CSV_FILE_PATH` variable is updated to point to the new file.
    *   The Pandas DataFrame and Langchain agent are re-initialized with the new CSV data.
4.  **Chat Interface:**
    *   The `/chat` route handles user queries.
    *   It receives a `user_query` from the user and passes it to the Langchain agent.
    *   The agent uses the Gemini LLM to generate a response based on the CSV data.
    *   The response is returned as a JSON object.
5.  **Deployment:**
    *   The `vercel.json` file configures the application for deployment on Vercel.
    *   It specifies that the `main.py` file should be used as the entry point for the application.
    *   It also defines a route that directs all requests to the `main.py` file.

## Dependencies

The application relies on the following Python libraries:

*   `fastapi`: A modern, fast (high-performance), web framework for building APIs with Python.
*   `langchain_experimental`: An experimental library that provides tools for building language model applications.
*   `langchain_google_genai`: A library that provides integration with Google's Gemini LLM.
*   `pandas`: A library for data manipulation and analysis.
*   `python-dotenv`: A library for loading environment variables from a `.env` file.
*   `uvicorn`: An ASGI server for running FastAPI applications.

## Configuration

The application can be configured using environment variables and the `credentials.json` file. The following environment variables are used:

*   `GEMINI_API_KEY`: The API key for accessing the Google Gemini LLM.
*   `PORT`: The port on which the application should listen (defaults to 8000).

The `credentials.json` file stores user credentials for authentication.

## Notes

*   The application uses a hardcoded CSV file path (`CSV_FILE_PATH`). This could be made configurable in a future version.
*   The application stores session data in memory (`active_sessions`). This is not suitable for production environments and should be replaced with a persistent storage solution such as Redis or a database.
*   The application uses `allow_dangerous_code=True` when creating the Langchain agent. This should be carefully considered in production environments, as it could allow users to execute arbitrary code.


