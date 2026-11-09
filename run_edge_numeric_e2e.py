"""Numeric-forest latency, size, and end-to-end time from the raw record."""
import os
import re
import time
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_edge_numeric_e2e.csv")
BATCHES = (1, 32, 128)
N_WARMUP = 3
N_TRIALS = 20
DROP = {"Timestamp", "Label", "y", "log_text", "key", "attack_name"}


def stats(times):
    arr = np.asarray(times, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "sd_ms": float(arr.std(ddof=1)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def time_call(fn):
    for _ in range(N_WARMUP):
        fn()
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return stats(times)


def size_mb(obj):
    buf = BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.tell() / (1024 * 1024)


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


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
    elif isinstance(model, RandomForestClassifier):
        model.n_jobs = 1


def main():
    print("Loading cache...", flush=True)
    df = pd.read_pickle(os.path.join(ROOT, "cache_dedup_grouped.pkl"))
    y = df["y"].to_numpy()
    idx = np.arange(len(df))
    tr_i, te_i = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
    stemmer = PorterStemmer()
    stop = set(stopwords.words("english"))
    rows = []

    def add(row):
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(
            f"{row['kind']} {row['feature']} {row['model']} batch {row['batch']} "
            f"median {row['median_ms']:.3f} mean {row['mean_ms']:.3f} "
            f"p95 {row['p95_ms']:.3f} max {row['max_ms']:.3f}",
            flush=True,
        )

    cols = [c for c in df.columns if c not in DROP]
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imp.fit_transform(X.iloc[tr_i]))
    Xte = scaler.transform(imp.transform(X.iloc[te_i]))
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    print("Fitting numeric forest...", flush=True)
    rf.fit(Xtr, y[tr_i])
    force_one_core(rf)
    pred = rf.predict(Xte)
    f1 = float(f1_score(y[te_i], pred, average="macro"))
    sz = size_mb((imp, scaler, rf))
    print(f"numeric F1 {f1:.4f} size {sz:.3f} MB", flush=True)
    raw_te = X.iloc[te_i]
    for bs in BATCHES:
        st = time_call(lambda b=bs: rf.predict_proba(Xte[:b]))
        st.update(
            kind="inference", feature="Numeric-78", model="Random Forest",
            batch=bs, f1_macro=f1, size_mb=sz,
        )
        add(st)

        def numeric_e2e(b=bs, _imp=imp, _scaler=scaler, _rf=rf, _raw=raw_te):
            block = _raw.iloc[:b]
            mat = _scaler.transform(_imp.transform(block))
            _rf.predict_proba(mat)

        st = time_call(numeric_e2e)
        st.update(
            kind="end_to_end", feature="Numeric-78", model="Random Forest",
            batch=bs, f1_macro=f1, size_mb=sz,
        )
        add(st)

    text_te = df.iloc[te_i]
    ports = text_te["Dst Port"].to_numpy()
    protos = text_te["Protocol"].to_numpy()
    durs = text_te["Flow Duration"].to_numpy()
    fwds = text_te["Tot Fwd Pkts"].to_numpy()
    bwds = text_te["Tot Bwd Pkts"].to_numpy()

    def texts_for(b):
        out = []
        for i in range(b):
            dst = "wellknown" if ports[i] < 1024 else "dynamic"
            raw = f"dst_{dst} proto_{protos[i]} dur_{durs[i]} fwd_{fwds[i]} bwd_{bwds[i]}"
            out.append(preprocess_text(raw, stemmer, stop))
        return out

    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
    Xtr_w = word.fit_transform(df.iloc[tr_i]["log_text"])
    Xte_w = word.transform(df.iloc[te_i]["log_text"])
    ens = make_ensemble()
    print("Fitting word ensemble...", flush=True)
    ens.fit(Xtr_w, y[tr_i])
    force_one_core(ens)
    word_size = size_mb((word, ens))
    for bs in BATCHES:
        st = time_call(lambda b=bs: ens.predict_proba(Xte_w[:b]))
        st.update(
            kind="inference", feature="Lexical-Word", model="Ensemble",
            batch=bs, f1_macro=np.nan, size_mb=word_size,
        )
        add(st)

        def word_e2e(b=bs, _word=word, _ens=ens):
            _ens.predict_proba(_word.transform(texts_for(b)))

        st = time_call(word_e2e)
        st.update(
            kind="end_to_end", feature="Lexical-Word", model="Ensemble",
            batch=bs, f1_macro=np.nan, size_mb=word_size,
        )
        add(st)

    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
    Xtr_c = char.fit_transform(df.iloc[tr_i]["log_text"])
    Xte_c = char.transform(df.iloc[te_i]["log_text"])
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    print("Fitting character LR...", flush=True)
    lr.fit(Xtr_c, y[tr_i])
    char_size = size_mb((char, lr))
    for bs in BATCHES:
        st = time_call(lambda b=bs: lr.predict_proba(Xte_c[:b]))
        st.update(
            kind="inference", feature="Structural-Char", model="LR",
            batch=bs, f1_macro=np.nan, size_mb=char_size,
        )
        add(st)

        def char_e2e(b=bs, _char=char, _lr=lr):
            _lr.predict_proba(_char.transform(texts_for(b)))

        st = time_call(char_e2e)
        st.update(
            kind="end_to_end", feature="Structural-Char", model="LR",
            batch=bs, f1_macro=np.nan, size_mb=char_size,
        )
        add(st)

    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
