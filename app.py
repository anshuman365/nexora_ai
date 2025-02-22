from flask import Flask, request, jsonify from transformers import pipeline import nltk import torch import tensorflow as tf import sklearn

app = Flask(name)

Load NLP models

nltk.download('punkt') qa_pipeline = pipeline("question-answering")

@app.route('/') def home(): return jsonify({"message": "Welcome to Nexora AI API!"})

@app.route('/qa', methods=['POST']) def question_answering(): data = request.get_json() question = data.get("question") context = data.get("context")

if not question or not context:
    return jsonify({"error": "Question and context required!"}), 400

result = qa_pipeline({"question": question, "context": context})
return jsonify(result)

if name == 'main': app.run(debug=True)

