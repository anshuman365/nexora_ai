from flask import Flask, request, jsonify
from transformers import pipeline
import nltk
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Disable GPU usage for TensorFlow (ensure only CPU is used)
tf.config.set_visible_devices([], 'GPU')

# Load NLP models (using a smaller model)
nltk.download('punkt')
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Route for the home page
@app.route('/')
def home():
    return jsonify({"message": "Welcome to Nexora AI API!"})

# Route for question answering
@app.route('/qa', methods=['POST'])
def question_answering():
    # Get data from the request
    data = request.get_json()

    # Get the question and context from the JSON payload
    question = data.get("question")
    context = data.get("context")

    # Check if question and context are provided
    if not question or not context:
        return jsonify({"error": "Question and context required!"}), 400

    # Perform question answering using the preloaded model
    result = qa_pipeline({"question": question, "context": context})

    # Return the result as a JSON response
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)