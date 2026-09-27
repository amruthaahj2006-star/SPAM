# SpamShield AI — Intelligent SMS Threat Detection

CodSoft Machine Learning Internship – Task 4: Spam SMS Detection

This repository implements an end-to-end ML project to classify SMS messages as HAM (legitimate) or SPAM (unwanted/promotional/scam).

IMPORTANT:
- Use only the provided dataset at `data/spam.csv`. Do NOT download other datasets or create fake data.
- The expected dataset columns: `v1` (label), `v2` (message), plus several `Unnamed:` columns. The code will clean and rename them.

Project structure
```
spamshield-ai/
│
├── data/
│   └── spam.csv             # <-- Place your provided dataset here
│
├── models/
│   ├── spam_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.json
│
├── results/
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── analytics.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Quick start

1. Clone or copy the repository locally and make sure `data/spam.csv` is placed at `spamshield-ai/data/spam.csv`.

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

3. Train models and produce saved artifacts:
```bash
python -m src.train_model
```
This will:
- Load and clean `data/spam.csv`.
- Train 3 models: Multinomial Naive Bayes, Logistic Regression, Linear SVM (LinearSVC).
- Evaluate and compare models.
- Save the best model and TF-IDF vectorizer to the `models/` directory.
- Save model metadata and results to `results/`.

4. Launch the Streamlit app:
```bash
streamlit run app.py
```
Open the provided URL in your browser. Enter a message and press "Predict" to get a prediction (HAM/ SPAM), probability, and top indicative words.

Notes and design choices
- The code uses TF-IDF for text features.
- The best model is selected according to test accuracy; if similar accuracy ties occur, preference ordering is Logistic Regression > Multinomial NB > Linear SVC (ensures predict_proba availability when possible).
- The scripts are intentionally modular: preprocess, train, predict, analytics.

If you want me to push these files to a GitHub repository, provide the repository owner/name and I will create the files there.