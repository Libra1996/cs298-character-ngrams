"""Latency, serialized size, and traced memory for the text setups.

Batches are 1, 32, and 128 flow strings. Hashing is only in this script:
same character lengths, no stored vocabulary.
"""
import os
import re
import time
import tracemalloc
from io import BytesIO

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords", quiet=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZES = (1, 32, 128)
N_WARMUP = 3
N_TRIALS = 20

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


def serialize_log_balanced(row):
    dst_type = "wellknown" if row["Dst Port"] < 1024 else "dynamic"
    return (
        f"dst_{dst_type} proto_{row['Protocol']} "
        f"dur_{row['Flow Duration']} fwd_{row['Tot Fwd Pkts']} bwd_{row['Tot Bwd Pkts']}"
    )


def preprocess_text(text, stop_words, stemmer):
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


def serialized_size_mb(obj) -> float:
    buf = BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.tell() / (1024 * 1024)


def latency_ms(predict_fn, X, batch_size: int):
    n = X.shape[0]
    idx = np.arange(min(batch_size, n))
    xb = X[idx]
    for _ in range(N_WARMUP):
        predict_fn(xb)
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        predict_fn(xb)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return float(arr.mean()), float(np.percentile(arr, 95))


def peak_memory_mb(predict_fn, X) -> float:
    # Use a moderate batch so peak is measurable but still edge-like
    xb = X[: min(128, X.shape[0])]
    tracemalloc.start()
    predict_fn(xb)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


t0 = time.time()
file_path = next(
    (
        p
        for p in [
            os.path.join(ROOT, "02-14-2018.csv"),
        ]
        if os.path.isfile(p)
    ),
    None,
)
if file_path is None:
    raise FileNotFoundError("02-14-2018.csv not found")

print(f"Loading: {file_path}")
df = pd.read_csv(file_path)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["Label"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)

print("Serializing + preprocessing...")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
df["log_text"] = df.apply(serialize_log_balanced, axis=1)
df["log_text"] = df["log_text"].apply(lambda t: preprocess_text(t, stop_words, stemmer))
df.drop_duplicates(subset=["log_text", "Label"], inplace=True)
print(f"Rows after dedup: {len(df)}")

X_train, X_test, y_train, y_test = train_test_split(
    df["log_text"],
    df["Label"],
    test_size=0.3,
    stratify=df["Label"],
    random_state=42,
)
X_train_list = X_train.tolist()
X_test_list = X_test.tolist()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print("Building feature matrices...")
tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
hash_char = HashingVectorizer(
    analyzer="char", ngram_range=(3, 5), n_features=10000, alternate_sign=False
)

X_train_word = tfidf_word.fit_transform(X_train_list)
X_test_word = tfidf_word.transform(X_test_list)
X_train_char = tfidf_char.fit_transform(X_train_list)
X_test_char = tfidf_char.transform(X_test_list)
X_train_hash = hash_char.transform(X_train_list)
X_test_hash = hash_char.transform(X_test_list)

ent_train = pd.DataFrame([extract_entity_counts(t) for t in X_train_list])[FEATURE_COLS]
ent_test = pd.DataFrame([extract_entity_counts(t) for t in X_test_list])[FEATURE_COLS]
scaler = StandardScaler(with_mean=True)
ent_train_s = scaler.fit_transform(ent_train.astype(float))
ent_test_s = scaler.transform(ent_test.astype(float))

X_train_entity = csr_matrix(ent_train_s)
X_test_entity = csr_matrix(ent_test_s)
X_train_hybrid = hstack([X_train_char, X_train_entity])
X_test_hybrid = hstack([X_test_char, X_test_entity])

feature_sets = [
    ("Lexical-Word (TF-IDF 1-2)", X_train_word, X_test_word, tfidf_word),
    ("Structural-Char (TF-IDF 3-5)", X_train_char, X_test_char, tfidf_char),
    ("Semantic-Entity (regex counts)", X_train_entity, X_test_entity, None),
    ("Hybrid-Robust (char TF-IDF + entity)", X_train_hybrid, X_test_hybrid, tfidf_char),
    ("Char-Hashing (n_features=10000)", X_train_hash, X_test_hash, hash_char),
]

rows = []
for config, Xtr, Xte, featurizer in feature_sets:
    print(f"\n===== {config} =====")
    for name, model in make_models().items():
        print(f"  Training {name}...")
        t_fit = time.time()
        model.fit(Xtr, y_train)
        fit_s = time.time() - t_fit
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        # Size: model (+ vectorizer/scaler when part of deployable pipeline)
        pipeline = {"model": model}
        if featurizer is not None:
            pipeline["featurizer"] = featurizer
        if "Hybrid" in config or "Entity" in config:
            pipeline["entity_scaler"] = scaler
        size_mb = serialized_size_mb(pipeline)

        peak_mb = peak_memory_mb(model.predict_proba, Xte)
        for bs in BATCH_SIZES:
            mean_ms, p95_ms = latency_ms(model.predict_proba, Xte, bs)
            per_sample_ms = mean_ms / bs
            rows.append(
                {
                    "Dataset": "IDS2018",
                    "Feature Config": config,
                    "Model": name,
                    "Batch Size": bs,
                    "Latency mean (ms/batch)": mean_ms,
                    "Latency p95 (ms/batch)": p95_ms,
                    "Latency mean (ms/sample)": per_sample_ms,
                    "Model+featurizer size (MB)": size_mb,
                    "Peak inference memory (MB)": peak_mb,
                    "Train time (s)": fit_s,
                    "F1 (macro)": f1,
                    "AUC-ROC": auc,
                    "Feature dim": Xtr.shape[1],
                }
            )
            print(
                f"    batch={bs}: mean={mean_ms:.3f}ms p95={p95_ms:.3f}ms "
                f"size={size_mb:.2f}MB F1={f1:.4f}"
            )

results = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "results_step7_edge_metrics.csv")
results.to_csv(out_csv, index=False)

# Batch 32 is the row I compare in the thesis.
summary = results[results["Batch Size"] == 32].copy()
for col in [
    "Latency mean (ms/batch)",
    "Latency p95 (ms/batch)",
    "Latency mean (ms/sample)",
    "Model+featurizer size (MB)",
    "Peak inference memory (MB)",
    "Train time (s)",
    "F1 (macro)",
    "AUC-ROC",
]:
    summary[col] = summary[col].round(4)
sum_csv = os.path.join(ROOT, "results_step7_edge_metrics_batch32_rounded.csv")
summary.to_csv(sum_csv, index=False)

print("\n=== Step 7 — Edge metrics (batch=32 summary) ===")
print(
    summary[
        [
            "Feature Config",
            "Model",
            "Latency mean (ms/batch)",
            "Latency p95 (ms/batch)",
            "Model+featurizer size (MB)",
            "Peak inference memory (MB)",
            "F1 (macro)",
            "AUC-ROC",
        ]
    ].to_string(index=False)
)
print(f"Saved: {out_csv}")
print(f"Saved: {sum_csv}")
print(f"Total time: {time.time() - t0:.1f}s")
