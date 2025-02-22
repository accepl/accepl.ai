from flask import Flask, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Example AI model - Replace with your actual model code
# You may load the model here using libraries like TensorFlow, PyTorch, or custom ones
# For now, this is a placeholder function for demonstration purposes
def get_ai_response(user_input):
    # Replace this function with your actual model inference code
    # For example: result = your_model.predict(user_input)
    return f"This is a response to your input: {user_input}"

# API route for handling user input and returning AI responses
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input from the request
    data = request.get_json()
    user_input = data.get('input', '')

    # Ensure input is not empty
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Get AI response based on the user input
    response = get_ai_response(user_input)

    # Return the prediction as a JSON response
    return jsonify({'prediction': response})

# Dynamic port for deployment (works for Render)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Get the port from Render environment or default to 8000
    app.run(debug=True, host="0.0.0.0", port=port)  # Bind to 0.0.0.0 for accessibility
