import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from inference import get_predictor, predict as predict_mood

app = Flask(__name__)

# Restrict CORS to an explicit allowlist instead of "*". Origins can be
# overridden via the ALLOWED_ORIGINS env var (comma-separated). Defaults cover
# the common local dev setups for serving the frontend.
_default_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", ",".join(_default_origins)).split(",")
    if o.strip()
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# Build the predictor at startup (transformer if available, else sklearn).
_predictor = get_predictor()
app.logger.info("Mood predictor backend: %s", _predictor.backend)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "text" not in data:
        return jsonify({"error": "Request body must be JSON with a 'text' field."}), 400

    text = data["text"]
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string."}), 400

    return jsonify({"mood": predict_mood(text)})


@app.errorhandler(Exception)
def handle_unexpected_error(err):
    # Never leak stack traces to clients, even if debug were on.
    app.logger.exception("Unhandled error during request")
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    # Debug defaults to OFF and must be explicitly opted into via env.
    # Bind to localhost by default; override HOST=0.0.0.0 only for trusted use.
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=debug, host=host, port=port)
