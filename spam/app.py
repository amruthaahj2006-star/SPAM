"""
Streamlit app for SpamShield AI.
Run with:
    streamlit run app.py
"""
import streamlit as st
from src.predict import load_model_and_vectorizer, predict_message
from src.analytics import top_n_features_for_class
import joblib
import os
import json
import pandas as pd

st.set_page_config(page_title="SpamShield AI", layout="wide")

st.title("SpamShield AI — SMS Threat Detection")
st.markdown("Classify SMS messages into HAM (legitimate) or SPAM (scam/promotional).")

# Load model and metadata
model, vectorizer, metadata = load_model_and_vectorizer()

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Predict SMS")
    message = st.text_area("Enter SMS message here:", height=180, value="")
    if st.button("Predict"):
        if message.strip() == "":
            st.warning("Please enter a message to classify.")
        else:
            result = predict_message(message)
            label_name = result["label_name"]
            prob = result["probability_spam"]
            if prob is not None:
                st.success(f"Prediction: {label_name} (P(spam) = {prob:.3f})")
            else:
                st.success(f"Prediction: {label_name}")
            # If metadata present, show model name
            if metadata:
                st.write("Model:", metadata.get("model_name", "Unknown"))
                st.write("Trained at:", metadata.get("trained_at", "Unknown"))

            # Show top indicative words
            try:
                top = top_n_features_for_class(model, vectorizer, n=15)
                st.subheader("Top indicative words")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("SPAM")
                    spam_df = pd.DataFrame(top["SPAM"], columns=["feature", "weight"])
                    st.table(spam_df)
                with c2:
                    st.write("HAM")
                    ham_df = pd.DataFrame(top["HAM"], columns=["feature", "weight"])
                    st.table(ham_df)
            except Exception as e:
                st.info("Unable to compute top features: " + str(e))

with col2:
    st.header("Model & Results")
    if metadata:
        st.write("Model name:", metadata.get("model_name", "Unknown"))
        st.write("Accuracy:", metadata.get("accuracy", "Unknown"))
        st.write("Trained at:", metadata.get("trained_at", "Unknown"))
    # Show saved images if present
    if os.path.exists("results/confusion_matrix.png"):
        st.image("results/confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
    if os.path.exists("results/model_comparison.png"):
        st.image("results/model_comparison.png", caption="Model Comparison", use_column_width=True)

    st.markdown("---")
    st.markdown("Developed by: SpamShield AI")
    st.markdown("Technologies: Python, Pandas, Scikit-learn, TF-IDF, MultinomialNB, LogisticRegression, LinearSVC, Streamlit")