"""Step 4: Hybrid-Robust — char TF-IDF (3-5) + regex entity counts."""
import os
import re
import time

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
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
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords", quiet=True)

ROOT = os.path.dirname(os.path.abspath(__file__))

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
    "count_hash": re.compile(
        r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b"
    ),
    "count_path": re.compile(
        r"(?:[A-Za-z]:\\(?:[^\s\\/:*?\"<>|\r\n]+\\)*[^\s\\/:*?\"<>|\r\n]+)"
        r"|(?:/(?:(?:etc|var|usr|home|tmp|opt|bin|sbin)/|\S+/)+)\S+"
    ),
}
FEATURE_COLS = list(ENTITY_PATTERNS.keys())


def extract_entity_counts(text: str) -> dict:
    return {name: len(pat.findall(text)) for name, pat in ENTITY_PATTERNS.items()}


def make_models():
    return {
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


def eval_hybrid(texts, labels, dataset_name: str) -> pd.DataFrame:
    print(f"\n========== Hybrid on {dataset_name} ==========")
    entity_df = pd.DataFrame(list(pd.Series(texts).astype(str).apply(extract_entity_counts)))
    print("Regex entity means:")
    print(entity_df[FEATURE_COLS].mean())
    print(
        "Rows with any entity > 0:",
        int((entity_df[FEATURE_COLS].sum(axis=1) > 0).sum()),
        "/",
        len(texts),
    )

    X_train_txt, X_test_txt, y_train, y_test, ent_train, ent_test = train_test_split(
        texts,
        labels,
        entity_df[FEATURE_COLS],
        test_size=0.3,
        stratify=labels,
        random_state=42,
    )

    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
    X_train_char = tfidf.fit_transform(X_train_txt)
    X_test_char = tfidf.transform(X_test_txt)
    print(f"Char TF-IDF vocab: {len(tfidf.vocabulary_)}")

    scaler = StandardScaler(with_mean=True)
    ent_train_s = scaler.fit_transform(ent_train.astype(float))
    ent_test_s = scaler.transform(ent_test.astype(float))

    X_train = hstack([X_train_char, csr_matrix(ent_train_s)])
    X_test = hstack([X_test_char, csr_matrix(ent_test_s)])
    print(f"Hybrid shape train: {X_train.shape}")

    rows = []
    for name, model in make_models().items():
        print(f"\nTraining {name}...")
        t = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"Train+predict: {time.time() - t:.1f}s")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC: {auc:.4f}")
        rows.append(
            {
                "Dataset": dataset_name,
                "Feature Config": "Hybrid-Robust (char TF-IDF + entity)",
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision (macro)": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "Recall (macro)": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "F1 (macro)": f1_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "AUC-ROC": auc,
            }
        )
    return pd.DataFrame(rows)


# ----- IDS2018 -----
t0 = time.time()
ids_path = next(
    (
        p
        for p in [
            os.path.join(ROOT, "02-14-2018.csv"),
        ]
        if os.path.isfile(p)
    ),
    None,
)
if ids_path is None:
    raise FileNotFoundError("02-14-2018.csv not found")

print(f"Loading IDS2018: {ids_path}")
df = pd.read_csv(ids_path)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["Label"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)


def serialize_log_balanced(row):
    dst_type = "wellknown" if row["Dst Port"] < 1024 else "dynamic"
    return (
        f"dst_{dst_type} proto_{row['Protocol']} "
        f"dur_{row['Flow Duration']} fwd_{row['Tot Fwd Pkts']} bwd_{row['Tot Bwd Pkts']}"
    )


print("Serializing IDS2018...")
df["log_text"] = df.apply(serialize_log_balanced, axis=1)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


df["log_text"] = df["log_text"].apply(preprocess_text)
df.drop_duplicates(subset=["log_text", "Label"], inplace=True)
print(f"IDS2018 rows after dedup: {len(df)}")

results_ids = eval_hybrid(df["log_text"].tolist(), df["Label"], "IDS2018")

# ----- CTI -----
cti_csv = os.path.join(ROOT, "data", "cti", "cyber_threat_intelligence.csv")
cti = pd.read_csv(cti_csv)
cti["text"] = cti["text"].fillna("").astype(str)
IOC_GOLD = {
    "IPV4",
    "URL",
    "DOMAIN",
    "MD5",
    "SHA1",
    "SHA2",
    "FILEPATH",
    "REGISTRYKEY",
    "EMAIL",
    "malware",
    "vulnerability",
    "threat-actor",
    "attack-pattern",
    "campaign",
    "tools",
}


def gold_ioc_label(entity_labels) -> int:
    if entity_labels is None or (
        isinstance(entity_labels, float) and np.isnan(entity_labels)
    ):
        return 0
    labels = set(str(entity_labels).split("|")) - {""}
    return int(len(labels & IOC_GOLD) > 0)


cti["Label"] = cti["entity_labels"].apply(gold_ioc_label)
print("CTI label distribution:")
print(cti["Label"].value_counts())

results_cti = eval_hybrid(cti["text"].tolist(), cti["Label"], "CTI (mrmoor)")

results = pd.concat([results_ids, results_cti], ignore_index=True)
out_csv = os.path.join(ROOT, "results_step4_hybrid_robust.csv")
results.to_csv(out_csv, index=False)
print("\n=== Hybrid-Robust — Summary ===")
print(results.round(4).to_string(index=False))
print(f"Saved: {out_csv}")

# Update combined ablation table
partial_path = os.path.join(ROOT, "results_steps1to4_ablation_partial.csv")
frames = []
if os.path.isfile(partial_path):
    prev = pd.read_csv(partial_path)
    # drop older hybrid rows if re-run
    if "Feature Config" in prev.columns:
        prev = prev[
            ~prev["Feature Config"].astype(str).str.contains("Hybrid", na=False)
        ]
    frames.append(prev)
frames.append(results)
combined = pd.concat(frames, ignore_index=True)
combined.to_csv(partial_path, index=False)
print(f"Updated: {partial_path}")
print(f"Total time: {time.time() - t0:.1f}s")
