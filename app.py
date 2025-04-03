from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load trained model and vectorizer
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer file not found. Train the model first.")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detector API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "No 'text' key found in request"}), 400

    input_text = [data["text"]]
    input_vectorized = vectorizer.transform(input_text)

    prediction = model.predict(input_vectorized)[0]
    result = "Real" if prediction == 1 else "Fake"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
