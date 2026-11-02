"""Step 2: Structural-Char — char TF-IDF (3-5) + LR, RF, soft-voting."""
import os
import re
import time

import nltk
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


# Same text corpus as Step 1 so only the vectorizer changes
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

print("Fitting Structural-Char TF-IDF (char 3-5, max 10k)...")
t2 = time.time()
tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=10000,
)
X_train_char = tfidf_char.fit_transform(X_train)
X_test_char = tfidf_char.transform(X_test)
print(f"Vocabulary size: {len(tfidf_char.vocabulary_)}")
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
    model.fit(X_train_char, y_train)
    y_pred = model.predict(X_test_char)
    y_proba = model.predict_proba(X_test_char)[:, 1]
    print(f"Train+predict: {time.time() - t3:.1f}s")
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    rows.append(
        {
            "Feature Config": "Structural-Char (TF-IDF 3-5)",
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro"),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro"),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
            "AUC-ROC": roc_auc_score(y_test, y_proba),
        }
    )

results_char = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "results_step2_structural_char.csv")
results_char.to_csv(out_csv, index=False)
print("\n=== Structural-Char — Summary ===")
print(results_char.round(4).to_string(index=False))
print(f"Saved: {out_csv}")

# Side-by-side with Step 1 if available
word_csv = os.path.join(ROOT, "results_step1_lexical_word_baseline.csv")
if os.path.isfile(word_csv):
    combined = pd.concat(
        [pd.read_csv(word_csv), results_char], ignore_index=True
    )
    combined_path = os.path.join(ROOT, "results_steps1to4_ablation_partial.csv")
    combined.to_csv(combined_path, index=False)
    print("\n=== Word vs Char (so far) ===")
    print(combined.round(4).to_string(index=False))
    print(f"Saved: {combined_path}")

print(f"Total time: {time.time() - t0:.1f}s")
