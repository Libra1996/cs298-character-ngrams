"""Step 1: Lexical-Word baseline — LR, RF, soft-voting ensemble."""
import os
import re
import time

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(ROOT, "02-14-2018.csv"),
    os.path.join(os.getcwd(), "02-14-2018.csv"),
]
file_path = next((p for p in _candidates if os.path.isfile(p)), None)
if file_path is None:
    raise FileNotFoundError("02-14-2018.csv not found")

print(f"Loading: {file_path}")
t0 = time.time()
df = pd.read_csv(file_path)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["Label"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)
print(f"Dataset loaded. Total records: {len(df)}")
print(df["Label"].value_counts())
print(f"Load+clean: {time.time() - t0:.1f}s")


def serialize_log_balanced(row):
    dst_type = "wellknown" if row["Dst Port"] < 1024 else "dynamic"
    return (
        f"dst_{dst_type} proto_{row['Protocol']} "
        f"dur_{row['Flow Duration']} fwd_{row['Tot Fwd Pkts']} bwd_{row['Tot Bwd Pkts']}"
    )


print("Serializing flows...")
t1 = time.time()
df["log_text"] = df.apply(serialize_log_balanced, axis=1)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


df["log_text"] = df["log_text"].apply(preprocess_text)
df.drop_duplicates(subset=["log_text", "Label"], inplace=True)
print(f"After text dedup: {len(df)} rows")
print(f"Serialize+preprocess: {time.time() - t1:.1f}s")

X_train, X_test, y_train, y_test = train_test_split(
    df["log_text"],
    df["Label"],
    test_size=0.3,
    stratify=df["Label"],
    random_state=42,
)

print("Fitting word TF-IDF...")
t2 = time.time()
tfidf_baseline = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
X_train_lexical = tfidf_baseline.fit_transform(X_train)
X_test_lexical = tfidf_baseline.transform(X_test)
print(f"Vocabulary size: {len(tfidf_baseline.vocabulary_)}")
print(f"TF-IDF: {time.time() - t2:.1f}s")

models = {
    "Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
    ),
    "Soft-Voting Ensemble": VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
                ),
            ),
        ],
        voting="soft",
    ),
}

rows = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    t3 = time.time()
    model.fit(X_train_lexical, y_train)
    y_pred = model.predict(X_test_lexical)
    y_proba = model.predict_proba(X_test_lexical)[:, 1]
    print(f"Train+predict: {time.time() - t3:.1f}s")
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    rows.append(
        {
            "Feature Config": "Lexical-Word (TF-IDF 1-2)",
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro"),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro"),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
            "AUC-ROC": roc_auc_score(y_test, y_proba),
        }
    )

results_word = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "results_step1_lexical_word_baseline.csv")
results_word.to_csv(out_csv, index=False)
print("\n=== Lexical-Word Baseline — Summary ===")
print(results_word.round(4).to_string(index=False))
print(f"Saved: {out_csv}")
print(f"Total time: {time.time() - t0:.1f}s")
