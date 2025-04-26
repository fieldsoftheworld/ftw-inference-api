"""
Fields of the World Inference API Server

This script starts the FastAPI application server with the specified configuration.
"""

import argparse
import os

import uvicorn
from app.core.config import get_settings

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run Fields of the World Inference API Server"
    )
    parser.add_argument("--host", type=str, help="Host address to bind server to")
    parser.add_argument("--port", type=int, help="Port to run server on")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Parse arguments
    args = parser.parse_args()

    # Set environment variable for config file if specified
    if args.config:
        config_path = os.path.abspath(args.config)
        if os.path.exists(config_path):
            os.environ["CONFIG_FILE"] = config_path

    # Get settings
    settings = get_settings()

    # Override settings with command line arguments
    host = args.host or settings.host
    port = args.port or settings.port
    debug = args.debug or settings.debug

    # Run server
    uvicorn.run("app.main:app", host=host, port=port, reload=debug)
