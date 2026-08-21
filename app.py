"""
Signal Trace — Next Word Predictor
Flask backend that loads the trained LSTM and SimpleRNN models plus the
shared tokenizer, and serves a small JSON API the frontend calls to predict
the next word(s) for a piece of seed text.
"""

import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load models + artifacts once, at startup
# ---------------------------------------------------------------------------
print("Loading tokenizer and sequence length...")
with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
with open(os.path.join(MODELS_DIR, "max_sequence_len.pkl"), "rb") as f:
    max_sequence_len = pickle.load(f)

print("Loading LSTM model...")
lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_next_word_model.h5"))

print("Loading RNN model...")
rnn_model = load_model(os.path.join(MODELS_DIR, "rnn_next_word_model.h5"))

INDEX_TO_WORD = {index: word for word, index in tokenizer.word_index.items()}
VOCAB_SIZE = len(tokenizer.word_index) + 1

MODELS = {"lstm": lstm_model, "rnn": rnn_model}

print("Ready. Vocabulary size:", VOCAB_SIZE, "| Max sequence length:", max_sequence_len)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def top_k_predictions(seed_text: str, model, k: int = 3):
    """Return the top-k (word, probability) predictions for the next token."""
    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    probs = model.predict(token_list, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:k]
    results = []
    for idx in top_indices:
        word = INDEX_TO_WORD.get(idx, "")
        if not word or word == "<OOV>":
            continue
        results.append({"word": word, "probability": float(probs[idx])})
    return results


def generate_sequence(seed_text: str, model, steps: int):
    """Greedily extend seed_text by `steps` words, returning each step's
    chosen word plus the runner-up candidates, for both models to compare."""
    text = seed_text
    trace = []
    for _ in range(steps):
        candidates = top_k_predictions(text, model, k=3)
        if not candidates:
            break
        chosen = candidates[0]["word"]
        trace.append({"word": chosen, "candidates": candidates})
        text += " " + chosen
    return text, trace


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        vocab_size=VOCAB_SIZE,
        max_sequence_len=max_sequence_len,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    seed_text = (data.get("text") or "").strip()
    steps = int(data.get("steps", 1))
    steps = max(1, min(steps, 15))

    if not seed_text:
        return jsonify({"error": "Type some seed text first."}), 400

    response = {}
    for key, model in MODELS.items():
        full_text, trace = generate_sequence(seed_text, model, steps)
        response[key] = {
            "full_text": full_text,
            "trace": trace,
        }

    return jsonify(response)


@app.route("/api/info")
def api_info():
    return jsonify(
        {
            "vocab_size": VOCAB_SIZE,
            "max_sequence_len": max_sequence_len,
            "models": list(MODELS.keys()),
        }
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
