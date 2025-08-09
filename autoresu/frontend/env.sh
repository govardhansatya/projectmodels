#!/bin/sh

# Environment configuration script for React frontend
# This script replaces environment variables in the built files

set -e

# Default values
REACT_APP_API_URL=${REACT_APP_API_URL:-http://localhost:8000}
REACT_APP_ENVIRONMENT=${REACT_APP_ENVIRONMENT:-production}

echo "Setting up environment variables..."
echo "REACT_APP_API_URL: $REACT_APP_API_URL"
echo "REACT_APP_ENVIRONMENT: $REACT_APP_ENVIRONMENT"

# Replace environment variables in JavaScript files
find /usr/share/nginx/html -name "*.js" -exec sed -i \
    -e "s|REACT_APP_API_URL_PLACEHOLDER|$REACT_APP_API_URL|g" \
    -e "s|REACT_APP_ENVIRONMENT_PLACEHOLDER|$REACT_APP_ENVIRONMENT|g" \
    {} \;

echo "Environment variables configured successfully"
