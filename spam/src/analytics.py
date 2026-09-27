"""
Analytics helpers for plotting and inspecting model coefficients.
"""
import os
from typing import List, Tuple

import joblib
import numpy as np
import matplotlib.pyplot as plt

MODELS_DIR = "models"


def top_n_features_for_class(model, vectorizer, n=20) -> dict:
    """
    For linear models (LogisticRegression, LinearSVC) and MultinomialNB,
    compute top n features for each class.
    Returns {"HAM": [(feature, weight), ...], "SPAM": [...]}
    """
    feature_names = vectorizer.get_feature_names_out()
    top_features = {"HAM": [], "SPAM": []}

    # LogisticRegression and LinearSVC: coef_ shape (1, n_features) for binary
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        # Positive coefs indicate higher likelihood of class 1 (SPAM)
        top_spam_idx = np.argsort(coefs)[-n:][::-1]
        top_ham_idx = np.argsort(coefs)[:n]
        top_features["SPAM"] = [(feature_names[i], float(coefs[i])) for i in top_spam_idx]
        top_features["HAM"] = [(feature_names[i], float(coefs[i])) for i in top_ham_idx]
        return top_features

    # MultinomialNB: feature_log_prob_ shape (n_classes, n_features)
    if hasattr(model, "feature_log_prob_"):
        log_prob = model.feature_log_prob_
        spam_idx = np.argsort(log_prob[1])[-n:][::-1]
        ham_idx = np.argsort(log_prob[0])[-n:][::-1]
        top_features["SPAM"] = [(feature_names[i], float(log_prob[1][i])) for i in spam_idx]
        top_features["HAM"] = [(feature_names[i], float(log_prob[0][i])) for i in ham_idx]
        return top_features

    return top_features