import joblib
import os
import sys

import warnings
warnings.filterwarnings("ignore")


# Find the root directory of the project automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model + vectorizer using absolute paths
model = joblib.load(os.path.join(BASE_DIR, "models", "mood_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

# Read input text
text = " ".join(sys.argv[1:])
vec = vectorizer.transform([text])

# Predict mood
prediction = model.predict(vec)[0]

print("Detected Mood:", prediction)
