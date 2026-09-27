"""
Train script for SpamShield AI.

Usage:
    python -m src.train_model
This will:
- Clean the dataset
- Split into train/test
- Fit a TF-IDF vectorizer
- Train three models: MultinomialNB, LogisticRegression, LinearSVC
- Evaluate, plot, and save the best model and vectorizer
- Save metadata to models/model_metadata.json
"""
import os
import json
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

from src.preprocess import load_data, clean_data

RESULTS_DIR = "results"
MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "spam.csv")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_model_comparison(results: dict, out_path: str):
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=accuracies, y=names, palette="viridis")
    plt.xlim(0, 1)
    plt.xlabel("Accuracy")
    plt.title("Model Comparison (Accuracy)")
    for i, v in enumerate(accuracies):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train():
    print("Loading data from:", DATA_PATH)
    raw = load_data(DATA_PATH)
    df = clean_data(raw)
    print(f"Dataset size after cleaning: {len(df)} rows")

    X = df["message"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training models...")
    models = {}
    # Multinomial Naive Bayes
    mnb = MultinomialNB()
    mnb.fit(X_train_tfidf, y_train)
    models["MultinomialNB"] = mnb

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, solver="liblinear")
    logreg.fit(X_train_tfidf, y_train)
    models["LogisticRegression"] = logreg

    # Linear SVM (LinearSVC)
    svc = LinearSVC(max_iter=5000)
    svc.fit(X_train_tfidf, y_train)
    models["LinearSVC"] = svc

    print("Evaluating models...")
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"accuracy": acc, "report": cr, "confusion_matrix": cm}
        print(f"- {name}: accuracy={acc:.4f}")

    # Save comparison plot
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plot_model_comparison(results, comp_path)
    print("Saved model comparison to", comp_path)

    # Select best model by accuracy. Tie-breaker preference: LogisticRegression > MultinomialNB > LinearSVC
    sorted_models = sorted(results.items(), key=lambda x: (x[1]["accuracy"], x[0]), reverse=True)
    # Reorder tie-breaker manually
    pref = ["LogisticRegression", "MultinomialNB", "LinearSVC"]
    best_name = sorted(sorted_models, key=lambda x: (x[1]["accuracy"], -pref.index(x[0]) if x[0] in pref else -100))[0][0]
    best_model = models[best_name]
    print(f"Selected best model: {best_name} with accuracy {results[best_name]['accuracy']:.4f}")

    # Save confusion matrix of best model
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(results[best_name]["confusion_matrix"], labels=["HAM", "SPAM"], out_path=cm_path)
    print("Saved confusion matrix to", cm_path)

    # Persist model and vectorizer
    model_path = os.path.join(MODELS_DIR, "spam_model.pkl")
    vect_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vect_path)
    print("Saved model to", model_path)
    print("Saved TF-IDF vectorizer to", vect_path)

    metadata = {
        "model_name": best_name,
        "accuracy": results[best_name]["accuracy"],
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "vectorizer": {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "stop_words": "english",
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to", metadata_path)

    print("Training complete.")


if __name__ == "__main__":
    ensure_dirs()
    train()