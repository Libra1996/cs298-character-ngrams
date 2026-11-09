"""Batch-32 medians, and 20 trials of the word ensemble at batch 128."""
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
OUT = os.path.join(ROOT, "results_edge_median_sitting.csv")
TRIALS_OUT = os.path.join(ROOT, "results_edge_median_trials.csv")
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
        for name, est in model.estimators:
            if name == "rf":
                est.n_jobs = 1
        for est in model.estimators_:
            if isinstance(est, RandomForestClassifier):
                est.n_jobs = 1


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


def time_batch(predict_fn, X, batch_size):
    xb = X[: min(batch_size, X.shape[0])]
    for _ in range(N_WARMUP):
        predict_fn(xb)
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        predict_fn(xb)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=float)
    summary = {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "sd_ms": float(arr.std(ddof=1)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }
    return summary, times


def main():
    print("Loading cache...", flush=True)
    df = pd.read_pickle(os.path.join(ROOT, "cache_dedup_grouped.pkl"))
    stemmer = PorterStemmer()
    stop = set(stopwords.words("english"))
    print("Stemming...", flush=True)
    text = [preprocess_text(t, stemmer, stop) for t in df["log_text"].tolist()]
    y = df["y"].to_numpy()
    X_train, X_test, y_train, _ = train_test_split(
        text, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"train {len(X_train)} test {len(X_test)}", flush=True)

    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
    hasher = HashingVectorizer(
        analyzer="char", ngram_range=(3, 5), n_features=10000, alternate_sign=False
    )
    print("Vectorizing...", flush=True)
    Xtr_w, Xte_w = word.fit_transform(X_train), word.transform(X_test)
    Xtr_c, Xte_c = char.fit_transform(X_train), char.transform(X_test)
    Xtr_h = hstack([Xtr_c, csr_matrix((Xtr_c.shape[0], 5))])
    Xte_h = hstack([Xte_c, csr_matrix((Xte_c.shape[0], 5))])
    Xtr_hash, Xte_hash = hasher.transform(X_train), hasher.transform(X_test)

    jobs = [
        ("Lexical-Word", "Ensemble", make_ensemble(), Xtr_w, Xte_w, True, (32, 128)),
        ("Structural-Char", "LR", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000), Xtr_c, Xte_c, False, (32,)),
        ("Structural-Char", "Ensemble", make_ensemble(), Xtr_c, Xte_c, True, (32,)),
        ("Hybrid-Robust", "Ensemble", make_ensemble(), Xtr_h, Xte_h, True, (32,)),
        ("Char-Hashing", "LR", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000), Xtr_hash, Xte_hash, False, (32,)),
    ]

    rows = []
    trial_rows = []
    for config, name, model, A, B, one_core, batches in jobs:
        print(f"Training {config} {name}...", flush=True)
        t0 = time.perf_counter()
        model.fit(A, y_train)
        print(f"  fit {time.perf_counter() - t0:.1f}s", flush=True)
        if one_core:
            force_one_core(model)
        for bs in batches:
            summary, times = time_batch(model.predict_proba, B, bs)
            summary.update({"feature": config, "model": name, "batch": bs})
            rows.append(summary)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            for i, ms in enumerate(times, start=1):
                trial_rows.append(
                    {"feature": config, "model": name, "batch": bs, "trial": i, "ms": ms}
                )
            pd.DataFrame(trial_rows).to_csv(TRIALS_OUT, index=False)
            print(
                f"  batch={bs}: median={summary['median_ms']:.3f} "
                f"mean={summary['mean_ms']:.3f} sd={summary['sd_ms']:.3f} "
                f"p95={summary['p95_ms']:.3f} max={summary['max_ms']:.3f}",
                flush=True,
            )
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
