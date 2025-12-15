#!/bin/bash
# Script to start the API server

echo "Starting Multi-Step Reasoning Agent API..."
echo "Make sure dependencies are installed: pip install -r requirements.txt"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set API key if provided as argument
if [ ! -z "$1" ]; then
    export GOOGLE_API_KEY="$1"
    echo "API key set from argument"
elif [ ! -z "$GOOGLE_API_KEY" ]; then
    echo "Using existing GOOGLE_API_KEY"
else
    echo "Warning: GOOGLE_API_KEY not set. Use mock=true in requests or set the environment variable."
fi

echo ""
echo "Starting server on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python3 api.py

