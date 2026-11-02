"""Retime the ensembles with the forest limited to one core."""
import os
import re
import time
import tracemalloc

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZES = (1, 32, 128)
N_WARMUP = 3
N_TRIALS = 20


def ensemble():
    return VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, n_jobs=1, random_state=42
                ),
            ),
        ],
        voting="soft",
    )


def latency_ms(predict_fn, X, batch_size):
    xb = X[: min(batch_size, X.shape[0])]
    for _ in range(N_WARMUP):
        predict_fn(xb)
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        predict_fn(xb)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return float(arr.mean()), float(np.percentile(arr, 95))


def peak_memory_mb(predict_fn, X):
    xb = X[: min(128, X.shape[0])]
    tracemalloc.start()
    predict_fn(xb)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


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
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["log_text"], df["y"], test_size=0.3, stratify=df["y"], random_state=42
)
Xtr, Xte = X_train.tolist(), X_test.tolist()
ytr = y_train.to_numpy()
print(f"rows {len(df)} train {len(Xtr)}", flush=True)

word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
Xtr_w, Xte_w = word.fit_transform(Xtr), word.transform(Xte)
Xtr_c, Xte_c = char.fit_transform(Xtr), char.transform(Xte)
Xtr_h = hstack([Xtr_c, csr_matrix((Xtr_c.shape[0], 5))])
Xte_h = hstack([Xte_c, csr_matrix((Xte_c.shape[0], 5))])

rows = []
for config, A, B in (
    ("Lexical-Word", Xtr_w, Xte_w),
    ("Structural-Char", Xtr_c, Xte_c),
    ("Hybrid-Robust", Xtr_h, Xte_h),
):
    print(f"Training {config} ensemble n_jobs=1...", flush=True)
    model = ensemble()
    t0 = time.time()
    model.fit(A, ytr)
    print(f"  fit {time.time() - t0:.1f}s", flush=True)
    peak = peak_memory_mb(model.predict_proba, B)
    for bs in BATCH_SIZES:
        mean_ms, p95_ms = latency_ms(model.predict_proba, B, bs)
        rows.append(
            {
                "Feature Config": config,
                "Model": "Soft-Voting Ensemble",
                "n_jobs": 1,
                "Batch Size": bs,
                "Latency mean (ms/batch)": mean_ms,
                "Latency p95 (ms/batch)": p95_ms,
                "Peak inference memory (MB)": peak,
            }
        )
        print(f"  batch={bs}: mean={mean_ms:.3f} p95={p95_ms:.3f} mem={peak:.4f}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results_edge_njobs1.csv"), index=False)
print("wrote results_edge_njobs1.csv", flush=True)
