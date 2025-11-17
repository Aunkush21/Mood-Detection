from flask_cors import CORS
import joblib
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE, "models", "mood_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE, "models", "vectorizer.pkl"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    vec = vectorizer.transform([data])
    pred = model.predict(vec)[0]
    return jsonify({"mood": pred})

app.run(debug=True, host="0.0.0.0")
