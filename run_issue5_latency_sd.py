"""Standard deviation of the 20 timed predict_proba trials.

Forests are fit with every core, then scored with one core. n_jobs does
not change the seed-42 trees.
"""
import os
import re
import time

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZES = (1, 32, 128)
N_WARMUP = 3
N_TRIALS = 20


def make_ensemble():
    return VotingClassifier(
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
    )


def force_one_core(model):
    if hasattr(model, "named_estimators_"):
        model.named_estimators_["rf"].n_jobs = 1
        # VotingClassifier also keeps the estimators list used at predict time.
        for name, est in model.estimators:
            if name == "rf":
                est.n_jobs = 1
        for est in model.estimators_:
            if isinstance(est, RandomForestClassifier):
                est.n_jobs = 1


def latency(predict_fn, X, batch_size):
    xb = X[: min(batch_size, X.shape[0])]
    for _ in range(N_WARMUP):
        predict_fn(xb)
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        predict_fn(xb)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return (
        float(arr.mean()),
        float(arr.std(ddof=1)),
        float(np.percentile(arr, 95)),
        float(arr.min()),
        float(arr.max()),
    )


print("Loading...", flush=True)
df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
df = df.drop_duplicates().dropna()
df["y"] = (df["Label"] != "Benign").astype(int)
dst = np.where(df["Dst Port"].to_numpy() < 1024, "wellknown", "dynamic")
raw = (
    "dst_" + pd.Series(dst, index=df.index)
    + " proto_" + df["Protocol"].astype(str)
    + " dur_" + df["Flow Duration"].astype(str)
    + " fwd_" + df["Tot Fwd Pkts"].astype(str)
    + " bwd_" + df["Tot Bwd Pkts"].astype(str)
)
stemmer = PorterStemmer()
stop = set(stopwords.words("english"))


def preprocess_text(text):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


df["log_text"] = raw.map(preprocess_text)
df = df.drop_duplicates(subset=["log_text", "y"])
X_train, X_test, y_train, _ = train_test_split(
    df["log_text"], df["y"], test_size=0.3, stratify=df["y"], random_state=42
)
Xtr, Xte = X_train.tolist(), X_test.tolist()
ytr = y_train.to_numpy()
print(f"rows {len(df)} train {len(Xtr)}", flush=True)

word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
hasher = HashingVectorizer(
    analyzer="char", ngram_range=(3, 5), n_features=10000, alternate_sign=False
)
Xtr_w, Xte_w = word.fit_transform(Xtr), word.transform(Xte)
Xtr_c, Xte_c = char.fit_transform(Xtr), char.transform(Xte)
Xtr_h = hstack([Xtr_c, csr_matrix((Xtr_c.shape[0], 5))])
Xte_h = hstack([Xte_c, csr_matrix((Xte_c.shape[0], 5))])
Xtr_hash, Xte_hash = hasher.transform(Xtr), hasher.transform(Xte)

jobs = [
    ("Lexical-Word", "Ensemble", make_ensemble(), Xtr_w, Xte_w, True),
    ("Structural-Char", "LR", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000), Xtr_c, Xte_c, False),
    ("Structural-Char", "Ensemble", make_ensemble(), Xtr_c, Xte_c, True),
    ("Hybrid-Robust", "Ensemble", make_ensemble(), Xtr_h, Xte_h, True),
    ("Char-Hashing", "LR", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000), Xtr_hash, Xte_hash, False),
]

rows = []
for config, name, model, A, B, one_core in jobs:
    print(f"Training {config} {name}...", flush=True)
    t0 = time.time()
    model.fit(A, ytr)
    print(f"  fit {time.time() - t0:.1f}s", flush=True)
    if one_core:
        force_one_core(model)
    for bs in BATCH_SIZES:
        mean_ms, sd_ms, p95_ms, min_ms, max_ms = latency(model.predict_proba, B, bs)
        rows.append(
            {
                "Feature Config": config,
                "Model": name,
                "Batch Size": bs,
                "mean_ms": mean_ms,
                "sd_ms": sd_ms,
                "p95_ms": p95_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "per_flow_ms": mean_ms / bs,
            }
        )
        print(
            f"  batch={bs}: mean={mean_ms:.3f} sd={sd_ms:.3f} p95={p95_ms:.3f} "
            f"min={min_ms:.3f} max={max_ms:.3f} per={mean_ms / bs:.4f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results_issue5_latency_sd.csv"), index=False)

print("done", flush=True)
