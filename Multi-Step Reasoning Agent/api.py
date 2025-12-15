from flask import Flask, request, jsonify
from interface import solve
import os

app = Flask(__name__)

# Enable CORS if available (optional)
try:
    from flask_cors import CORS
    CORS(app)  # Enable CORS for all routes
except ImportError:
    # CORS not installed, continue without it
    pass

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Multi-Step Reasoning Agent"})

@app.route('/solve', methods=['POST'])
def solve_endpoint():
    """
    Main endpoint to solve word problems.
    
    Expected JSON payload:
    {
        "question": "Your question here",
        "mock": false  // optional, defaults to false
    }
    
    Returns:
    {
        "answer": "...",
        "status": "success" | "failed",
        "reasoning_visible_to_user": "...",
        "metadata": {...}
    }
    """
    try:
        # Get JSON payload
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON payload provided",
                "status": "error"
            }), 400
        
        # Extract question
        question = data.get('question')
        if not question:
            return jsonify({
                "error": "Missing 'question' field in payload",
                "status": "error"
            }), 400
        
        # Extract mock flag (optional, defaults to False)
        mock = data.get('mock', False)
        
        # Check if API key is set (unless mock mode)
        if not mock and not os.getenv('GOOGLE_API_KEY'):
            return jsonify({
                "error": "GOOGLE_API_KEY environment variable not set. Use mock=true for testing.",
                "status": "error"
            }), 500
        
        # Solve the question
        result = solve(question, mock=mock)
        
        # Return the result
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API documentation endpoint"""
    return jsonify({
        "service": "Multi-Step Reasoning Agent API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "POST /solve": {
                "description": "Solve a word problem",
                "payload": {
                    "question": "string (required) - The word problem to solve",
                    "mock": "boolean (optional) - Use mock LLM, defaults to false"
                },
                "example": {
                    "question": "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?",
                    "mock": False
                }
            }
        }
    })

if __name__ == '__main__':
    # Run on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)

