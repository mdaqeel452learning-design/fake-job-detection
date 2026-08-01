"""Train the fake-job classifier and record real, measured accuracy.

Run from the web/ directory:
    python train_model.py
"""
import json
import os
import pickle
from datetime import datetime, timezone

import nltk
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from detection import METRICS_PATH, MODEL_PATH, VECTORIZER_PATH, preprocess_text

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "fake_job_postings.csv")

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)


def main():
    data = pd.read_csv(DATA_PATH)
    data["text"] = data["title"].fillna("") + " " + data["description"].fillna("")
    data["clean_text"] = data["text"].apply(preprocess_text)

    # Structural signals: by far the strongest fraud predictors in this
    # dataset (see README), but invisible to a text-only model. They mirror
    # what the web form now asks the user directly, so training and serving
    # use the exact same three columns.
    data["has_company_logo"] = data["has_company_logo"].fillna(0).astype(float)
    data["has_company_profile"] = (
        data["company_profile"].fillna("").str.strip().str.len() > 0
    ).astype(float)
    data["has_salary_range"] = (
        data["salary_range"].fillna("").str.strip().str.len() > 0
    ).astype(float)

    X_text = data["clean_text"]
    X_struct = data[["has_company_logo", "has_company_profile", "has_salary_range"]]
    y = data["fraudulent"].astype(int)

    # Held-out test set, stratified so the ~5% fraud rate is preserved in
    # both splits — this is what makes the reported accuracy trustworthy.
    X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
        X_text, X_struct, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
    X_text_train_vec = vectorizer.fit_transform(X_text_train)
    X_text_test_vec = vectorizer.transform(X_text_test)

    X_train = hstack([X_text_train_vec, X_struct_train.values]).tocsr()
    X_test = hstack([X_text_test_vec, X_struct_test.values]).tocsr()

    # The dataset is heavily imbalanced (~5% fraudulent). MLPClassifier has
    # no class_weight support, so balance the *training* set only via
    # oversampling the minority class; the test set stays untouched so the
    # evaluation reflects real-world class distribution.
    train_df = pd.DataFrame({"idx": range(X_train.shape[0]), "y": y_train.values})
    majority = train_df[train_df.y == 0]
    minority = train_df[train_df.y == 1]
    minority_upsampled = resample(
        minority, replace=True, n_samples=len(majority), random_state=42
    )
    balanced_idx = pd.concat([majority, minority_upsampled])["idx"].values
    X_train_bal = X_train[balanced_idx]
    y_train_bal = y_train.values[balanced_idx]

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": {"labels": ["Real", "Fake"], "matrix": cm},
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train_bal)),
        "features": "TF-IDF(title+description) + has_company_logo + has_company_profile + has_salary_range",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model trained and saved.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
