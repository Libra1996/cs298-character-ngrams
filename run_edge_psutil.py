"""Resident set for the five timed setups, each in a fresh process.

A fresh process keeps one model's memory from sitting on top of the last
model. The forests are scored with one core.
"""
import os
import re
import subprocess
import sys
import tempfile

import joblib
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


def measure_file(path):
    import gc
    import threading
    import time
    import tracemalloc

    import psutil

    model, X = joblib.load(path)
    gc.collect()
    proc = psutil.Process()
    model.predict_proba(X)
    rss_warm = proc.memory_info().rss
    peak = {"rss": rss_warm}
    stop = {"v": False}

    def sample():
        while not stop["v"]:
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            time.sleep(0.001)

    thread = threading.Thread(target=sample)
    thread.start()
    tracemalloc.start()
    for _ in range(20):
        model.predict_proba(X)
    _, traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stop["v"] = True
    thread.join()
    uss = getattr(proc.memory_full_info(), "uss", 0)
    print(f"rss_warm_mb {rss_warm / (1024 * 1024):.2f}")
    print(f"rss_peak_mb {peak['rss'] / (1024 * 1024):.2f}")
    print(f"uss_mb {uss / (1024 * 1024):.2f}")
    print(f"traced_mb {traced / (1024 * 1024):.4f}")


if len(sys.argv) > 1 and sys.argv[1] == "measure":
    measure_file(sys.argv[2])
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "baseline":
    import psutil

    rss = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"baseline_rss_mb {rss:.2f}")
    sys.exit(0)


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


def lr():
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)


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
X_train, X_test, y_train, y_test = train_test_split(
    df["log_text"], df["y"], test_size=0.3, stratify=df["y"], random_state=42
)
Xtr, Xte = X_train.tolist(), X_test.tolist()
ytr = y_train.to_numpy()
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

base = subprocess.run(
    [sys.executable, __file__, "baseline"], capture_output=True, text=True, check=True
)
print(base.stdout.strip(), flush=True)

jobs = [
    ("Lexical-Word", "Ensemble", ensemble(), Xtr_w, Xte_w[:128]),
    ("Structural-Char", "LR", lr(), Xtr_c, Xte_c[:128]),
    ("Structural-Char", "Ensemble", ensemble(), Xtr_c, Xte_c[:128]),
    ("Hybrid-Robust", "Ensemble", ensemble(), Xtr_h, Xte_h[:128]),
    ("Char-Hashing", "LR", lr(), Xtr_hash, Xte_hash[:128]),
]
rows = []
for config, name, model, A, B in jobs:
    print(f"Training {config} {name}...", flush=True)
    model.fit(A, ytr)
    fd, path = tempfile.mkstemp(suffix=".joblib")
    os.close(fd)
    joblib.dump((model, B), path, compress=3)
    proc = subprocess.run(
        [sys.executable, __file__, "measure", path],
        capture_output=True,
        text=True,
        check=True,
    )
    os.remove(path)
    print(proc.stdout, flush=True)
    vals = dict(line.split(" ", 1) for line in proc.stdout.strip().splitlines())
    rows.append(
        {
            "Feature Config": config,
            "Model": name,
            "RSS warmup MB": float(vals["rss_warm_mb"]),
            "RSS peak MB": float(vals["rss_peak_mb"]),
            "USS MB": float(vals["uss_mb"]),
            "Traced MB": float(vals["traced_mb"]),
        }
    )

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results_edge_psutil.csv"), index=False)
print(out.to_string(index=False), flush=True)
print("wrote results_edge_psutil.csv", flush=True)
