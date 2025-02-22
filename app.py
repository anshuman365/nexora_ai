from flask import Flask, request, jsonify
from transformers import pipeline
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download NLTK punkt
nltk.download('punkt')

# Load a suitable model for text generation (e.g., GPT2)
model = pipeline('text-generation', model='gpt2')  # or 'distilgpt2' for a lighter model

# Route for the home page
@app.route('/')
def home():
    return jsonify({"message": "Welcome to Nexora AI API!"})

# Route for text generation
@app.route('/generate', methods=['POST'])
def generate_text():
    # Get data from the request
    data = request.get_json()

    # Get the input text from the JSON payload
    prompt = data.get("prompt")

    # Check if prompt is provided
    if not prompt:
        return jsonify({"error": "Prompt is required!"}), 400

    # Perform text generation using the preloaded model
    result = model(prompt, max_length=100, num_return_sequences=1)

    # Return the generated text as a JSON response
    return jsonify(result[0])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)