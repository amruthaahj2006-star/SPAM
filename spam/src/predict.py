"""
Prediction utilities for SpamShield AI.
"""
import os
import json
from typing import Tuple, Dict

import joblib
import numpy as np

MODELS_DIR = "models"


def load_model_and_vectorizer(model_path: str = None, vectorizer_path: str = None):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "spam_model.pkl")
    if vectorizer_path is None:
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    # Try to load metadata if available
    metadata = {}
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    return model, vectorizer, metadata


def predict_message(message: str) -> Dict:
    """
    Predict whether a single SMS message is HAM (0) or SPAM (1).

    Returns a dict with:
    - label (0 or 1)
    - label_name ("HAM"/"SPAM")
    - probability (if available, else None)
    """
    model, vectorizer, metadata = load_model_and_vectorizer()
    X = vectorizer.transform([message])
    result = {}
    label = int(model.predict(X)[0])
    result["label"] = label
    result["label_name"] = "SPAM" if label == 1 else "HAM"

    # Try to obtain probability if model supports it
    prob = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        # For binary 0/1, probability of class 1 is probs[1]
        prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        # fallback: use decision_function scaled via sigmoid
        if hasattr(model, "decision_function"):
            try:
                score = float(model.decision_function(X)[0])
                prob = 1.0 / (1.0 + np.exp(-score))
            except Exception:
                prob = None
    result["probability_spam"] = prob

    # Include metadata
    result["metadata"] = metadata

    return result


if __name__ == "__main__":
    # Quick local test when run directly (assumes models/ exists)
    sample = "Free entry in 2 a weekly competition to win FA Cup final tickets. Text WIN to 12345"
    print("Predicting sample message:")
    print(predict_message(sample))