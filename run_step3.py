"""Step 3: Semantic-Entity — regex entity counts + LR, RF, soft-voting."""
import os
import re
import time

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
from sklearn.preprocessing import StandardScaler

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

# IP, URL or domain, CVE, hash (32, 40, or 64 hex digits), and file path.
ENTITY_PATTERNS = {
    "count_ip": re.compile(
        r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?!\d)"
    ),
    "count_url": re.compile(
        r"(?:https?://|www\.)[^\s]+|(?:[a-z0-9-]+\.)+(?:com|net|org|io|edu|gov)(?:/[^\s]*)?",
        re.IGNORECASE,
    ),
    "count_cve": re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE),
    "count_hash": re.compile(r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b"),
    "count_path": re.compile(
        r"(?:[A-Za-z]:\\(?:[^\s\\/:*?\"<>|\r\n]+\\)*[^\s\\/:*?\"<>|\r\n]+)"
        r"|(?:/(?:(?:etc|var|usr|home|tmp|opt|bin|sbin)/|\S+/)+)\S+"
    ),
}


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


def extract_entity_counts(text: str) -> dict:
    return {name: len(pat.findall(text)) for name, pat in ENTITY_PATTERNS.items()}


print("Extracting Semantic-Entity features (regex counts)...")
t2 = time.time()
entity_rows = df["log_text"].apply(extract_entity_counts)
entity_df = pd.DataFrame(list(entity_rows))
entity_df.index = df.index
feature_cols = list(ENTITY_PATTERNS.keys())
print(f"Entity extraction: {time.time() - t2:.1f}s")
print("\nEntity count means (expect near-zero on flow-stat-only text):")
print(entity_df[feature_cols].mean())
print("\nEntity count maxima:")
print(entity_df[feature_cols].max())
print(f"Rows with any entity > 0: {(entity_df[feature_cols].sum(axis=1) > 0).sum()}")

X_train_txt, X_test_txt, y_train, y_test, ent_train, ent_test = train_test_split(
    df["log_text"],
    df["Label"],
    entity_df[feature_cols],
    test_size=0.3,
    stratify=df["Label"],
    random_state=42,
)

# Scale for LR; RF is scale-insensitive but shared matrix is fine
scaler = StandardScaler(with_mean=True)
X_train_ent = scaler.fit_transform(ent_train.astype(float))
X_test_ent = scaler.transform(ent_test.astype(float))

# If all features are constant, StandardScaler yields 0; models use intercept/majority
nonzero_var = np.var(X_train_ent, axis=0).sum()
print(f"\nTrain feature variance sum: {nonzero_var:.6f}")
if nonzero_var == 0:
    print(
        "WARNING: Entity features have zero variance on this dataset. "
        "Semantic-Entity-only models cannot learn from regex IoCs here "
        "(IDS2018 flow CSV has no IP/URL/CVE/hash/path text)."
    )

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
    model.fit(X_train_ent, y_train)
    y_pred = model.predict(X_test_ent)
    # predict_proba may be constant when features are useless
    y_proba = model.predict_proba(X_test_ent)[:, 1]
    print(f"Train+predict: {time.time() - t3:.1f}s")
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")
    print(f"AUC-ROC: {auc:.4f}" if auc == auc else "AUC-ROC: nan")
    rows.append(
        {
            "Feature Config": "Semantic-Entity (regex counts)",
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall (macro)": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "AUC-ROC": auc,
        }
    )

results_entity = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "results_step3_semantic_entity_ids2018.csv")
results_entity.to_csv(out_csv, index=False)
print("\n=== Semantic-Entity — Summary ===")
print(results_entity.round(4).to_string(index=False))
print(f"Saved: {out_csv}")

diag = {
    "rows_total": len(df),
    "rows_with_any_entity": int((entity_df[feature_cols].sum(axis=1) > 0).sum()),
    **{f"mean_{c}": float(entity_df[c].mean()) for c in feature_cols},
    **{f"max_{c}": int(entity_df[c].max()) for c in feature_cols},
}
diag_path = os.path.join(ROOT, "results_step3_semantic_entity_ids2018_diagnostics.csv")
pd.DataFrame([diag]).to_csv(diag_path, index=False)
print(f"Saved: {diag_path}")

partial = os.path.join(ROOT, "results_steps1to4_ablation_partial.csv")
if os.path.isfile(partial):
    combined = pd.concat(
        [pd.read_csv(partial), results_entity], ignore_index=True
    )
else:
    combined = results_entity
combined_path = os.path.join(ROOT, "results_steps1to4_ablation_partial.csv")
combined.to_csv(combined_path, index=False)
print("\n=== Ablation so far ===")
print(combined.round(4).to_string(index=False))
print(f"Saved: {combined_path}")
print(f"Total time: {time.time() - t0:.1f}s")
