"""CLI mood predictor:  python src/predict.py "your text here" """

import sys
import warnings

warnings.filterwarnings("ignore")

from inference import get_predictor


def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print('Usage: python src/predict.py "text to analyze"')
        sys.exit(1)

    predictor = get_predictor()
    mood = predictor.predict(text)
    print(f"Detected Mood: {mood}  (model: {predictor.backend})")


if __name__ == "__main__":
    main()
