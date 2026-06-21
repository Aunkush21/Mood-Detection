# 🧠 MoodSense — AI Mood Detection from Text

A full-stack machine learning app that reads the **emotional tone** behind any
piece of text and classifies it into one of four moods:

😄 **Happy**  ·  😢 **Sad**  ·  😡 **Angry**  ·  😐 **Neutral**

It ships with a fine-tuned **DistilBERT transformer** for accuracy, a
lightweight **scikit-learn** fallback that works out of the box, a hardened
**Flask** API, and a brand-new **mood-reactive** web UI whose colors shift to
match the detected emotion.

---

## ✨ Features

- **Two models, one interface** — a fine-tuned DistilBERT transformer when
  available, automatically falling back to a TF-IDF + Logistic Regression model.
- **Trained on Google's GoEmotions** — all 27 fine-grained emotions mapped down
  to 4 moods with majority-vote multi-label resolution.
- **Mood-reactive UI** — the background aurora, glow, result orb and accent
  colors all recolor based on the prediction (happy → amber, sad → blue,
  angry → red, neutral → slate).
- **Hardened Flask API** — strict input validation, CORS allowlist, no debug
  exposure, generic error responses, and SHA-256 integrity checks on model
  files before loading.
- **Accessible & responsive** — keyboard support (`Ctrl`+`Enter`), focus states,
  `prefers-reduced-motion`, and mobile layouts.

---

## 📊 Model Performance

Evaluated on the GoEmotions **test split** (balanced 4-mood mapping):

| Mood        | scikit-learn F1 | DistilBERT F1 |
|-------------|:---------------:|:-------------:|
| happy       | 0.77            | **0.82**      |
| neutral     | 0.70            | **0.74**      |
| sad         | 0.56            | **0.66**      |
| angry       | 0.51            | **0.61**      |
| **Accuracy**    | 0.68        | **0.745**     |
| **Macro-F1**    | 0.63        | **0.708**     |

> Note: 4-way emotion classification is genuinely hard — macro-F1 is the honest
> metric here, since the neutral class is the most common.

---

## 🗂️ Project Structure

```
.
├── frontend/
│   ├── index.html        # MoodSense UI
│   ├── style.css         # mood-reactive theming
│   └── script.js         # calls /predict, themes the page
├── models/
│   ├── mood_model.pkl     # scikit-learn classifier (committed)
│   ├── vectorizer.pkl     # TF-IDF vectorizer (committed)
│   ├── model_hashes.json  # SHA-256 integrity hashes
│   └── transformer/       # DistilBERT model (generated locally, git-ignored)
├── src/
│   ├── api.py             # Flask API (POST /predict)
│   ├── inference.py       # shared predictor (transformer → sklearn fallback)
│   ├── predict.py         # CLI predictor
│   ├── train.py           # train the scikit-learn model
│   └── train_transformer.py  # fine-tune DistilBERT
├── notebook/             # original data-exploration notebook
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training/inference, install the CUDA build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. Run the API

```bash
# defaults: host 127.0.0.1, port 5000, debug OFF
python src/api.py
```

The API prefers `models/transformer/` if present, otherwise uses the bundled
scikit-learn model.

### 3. Serve the frontend

```bash
python -m http.server 5500 --directory frontend
```

Open **http://localhost:5500** and start typing. (Serve it via HTTP — opening
`index.html` directly as a `file://` URL is blocked by the API's CORS policy.)

---

## 🧪 Usage

### Web

Type a sentence, hit **Analyze mood** (or `Ctrl`+`Enter`), and watch the whole
interface recolor to the detected emotion.

### CLI

```bash
python src/predict.py "i can't stop smiling today"
# Detected Mood: happy  (model: transformer)
```

### API

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "this is the worst day ever"}'
# {"mood": "angry"}
```

---

## 🏋️ Training

```bash
# scikit-learn model (fast, CPU-friendly)
python src/train.py

# DistilBERT transformer (GPU recommended) — regenerates models/transformer/
python src/train_transformer.py
```

Both scripts download GoEmotions automatically, save the model + vectorizer/
tokenizer to `models/`, and write SHA-256 integrity hashes.

---

## 🔒 Security

- Flask debug mode is **off by default** (opt-in via `FLASK_DEBUG=1`) and binds
  to `127.0.0.1` unless `HOST` is overridden.
- `/predict` validates the request body and returns a generic `400`/`500`
  without leaking stack traces.
- CORS is restricted to an allowlist (override with the `ALLOWED_ORIGINS` env
  var).
- The frontend renders predictions via `textContent` against a fixed mood
  whitelist (no HTML injection).
- Model files are verified against recorded SHA-256 hashes before loading.

---

## ⚙️ Configuration

| Env var           | Default                        | Description                          |
|-------------------|--------------------------------|--------------------------------------|
| `HOST`            | `127.0.0.1`                    | API bind address                     |
| `PORT`            | `5000`                         | API port                             |
| `FLASK_DEBUG`     | `0`                            | Set `1` to enable debug (dev only)   |
| `ALLOWED_ORIGINS` | localhost:5500 / :3000         | Comma-separated CORS allowlist       |

---

## 🛠️ Tech Stack

Python · Flask · scikit-learn · PyTorch · Hugging Face Transformers (DistilBERT)
· GoEmotions · HTML / CSS / JavaScript
