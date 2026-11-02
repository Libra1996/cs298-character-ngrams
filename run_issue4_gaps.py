"""Depth sweep, linear SVM, duration encoding, dropping one field, and shuffled test tokens.

Seed 42 is the same split as the main tables. Seed 21 is only the word
forest, the trial that predicted all benign at depth 10.
"""
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_issue4_gaps.csv")
PREFIXES = ("dst_", "proto_", "dur_", "fwd_", "bwd_")


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


def duration_bin(d):
    if d <= 0:
        return "dur_0"
    if d < 1_000:
        return "dur_lt1ms"
    if d < 100_000:
        return "dur_lt100ms"
    if d < 1_000_000:
        return "dur_lt1s"
    if d < 10_000_000:
        return "dur_lt10s"
    if d < 60_000_000:
        return "dur_lt60s"
    return "dur_ge60s"


def drop_field(text, prefix):
    return " ".join(tok for tok in str(text).split() if not tok.startswith(prefix))


def reorder_tokens(texts, seed):
    rng = np.random.RandomState(seed)
    out = []
    for text in texts:
        toks = str(text).split()
        rng.shuffle(toks)
        out.append(" ".join(toks))
    return out


def pack(block, feature, model, y, pred, score):
    y = np.asarray(y)
    row = {
        "block": block,
        "feature": feature,
        "model": model,
        "n_test": int(len(y)),
        "n_pos": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y, score)) if y.sum() not in (0, len(y)) else 0.5,
        "attack_recall": float(((pred == 1) & (y == 1)).sum() / max(int(y.sum()), 1)),
        "pred_attack": int((pred == 1).sum()),
    }
    print(
        f"{block} {feature} {model} F1 {row['f1_macro']:.4f} "
        f"Acc {row['accuracy']:.4f} AUC {row['auc']:.4f} "
        f"atkR {row['attack_recall']:.4f} pred+ {row['pred_attack']}",
        flush=True,
    )
    return row


def fit_lr(Xtr, ytr, Xte, yte, block, feature):
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(Xtr, ytr)
    proba = lr.predict_proba(Xte)[:, 1]
    return pack(block, feature, "Logistic Regression", yte, (proba >= 0.5).astype(int), proba)


def fit_svm(Xtr, ytr, Xte, yte, block, feature):
    svm = LinearSVC(C=1.0, dual=False, max_iter=5000)
    svm.fit(Xtr, ytr)
    score = svm.decision_function(Xte)
    pred = svm.predict(Xte)
    return pack(block, feature, "Linear SVM", yte, pred, score)


def fit_rf(Xtr, ytr, Xte, yte, block, feature, depth, seed):
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(Xtr, ytr)
    proba = rf.predict_proba(Xte)[:, 1]
    label = "RF depth unlimited" if depth is None else f"RF depth {depth}"
    return pack(block, feature, label, yte, (proba >= 0.5).astype(int), proba)


def texts_xy(frame, column, train_idx, test_idx):
    return (
        frame.loc[train_idx, column],
        frame.loc[train_idx, "y"],
        frame.loc[test_idx, column],
        frame.loc[test_idx, "y"],
    )


def vectorize(train_text, test_text, analyzer, ngram):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=10000)
    return vec.fit_transform(train_text), vec.transform(test_text)


def save(rows):
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, "rows", len(rows), flush=True)


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
df["exact"] = raw.map(lambda t: preprocess_text(t, stemmer, stop))
df["dur"] = df["Flow Duration"].astype("int64")
df = df.drop_duplicates(subset=["exact", "y"]).copy()
df["binned"] = [
    re.sub(r"dur_\d+", duration_bin(d), text)
    for text, d in zip(df["exact"], df["dur"])
]
for prefix in PREFIXES:
    df["drop_" + prefix[:-1]] = df["exact"].map(lambda t, p=prefix: drop_field(t, p))

rows = []
train_idx, test_idx = train_test_split(
    df.index, test_size=0.3, stratify=df["y"], random_state=42
)
print("seed42 train", len(train_idx), "test", len(test_idx), flush=True)

# 2x2 duration and linear SVM on the same seed-42 rows.
for feature, column, analyzer, ngram in (
    ("word exact", "exact", "word", (1, 2)),
    ("char exact", "exact", "char", (3, 5)),
    ("word binned", "binned", "word", (1, 2)),
    ("char binned", "binned", "char", (3, 5)),
):
    tr, ytr, te, yte = texts_xy(df, column, train_idx, test_idx)
    Xtr, Xte = vectorize(tr, te, analyzer, ngram)
    rows.append(fit_lr(Xtr, ytr, Xte, yte, "duration_2x2", feature))
    if column == "exact":
        rows.append(fit_svm(Xtr, ytr, Xte, yte, "linear_svm", feature))
    save(rows)

# Drop one field at a time. Word and character logistic regression.
for prefix in PREFIXES:
    column = "drop_" + prefix[:-1]
    for feature, analyzer, ngram in (
        (f"word drop {prefix[:-1]}", "word", (1, 2)),
        (f"char drop {prefix[:-1]}", "char", (3, 5)),
    ):
        tr, ytr, te, yte = texts_xy(df, column, train_idx, test_idx)
        Xtr, Xte = vectorize(tr, te, analyzer, ngram)
        rows.append(fit_lr(Xtr, ytr, Xte, yte, "field_ablation", feature))
    save(rows)

# Reordered test tokens. Train on clean exact text.
for feature, analyzer, ngram in (
    ("word reorder", "word", (1, 2)),
    ("char reorder", "char", (3, 5)),
):
    tr, ytr, te, yte = texts_xy(df, "exact", train_idx, test_idx)
    te = reorder_tokens(te, seed=42)
    Xtr, Xte = vectorize(tr, te, analyzer, ngram)
    rows.append(fit_lr(Xtr, ytr, Xte, yte, "token_reorder", feature))
save(rows)

# Depth sweep, seed 42, word and character forests.
for feature, analyzer, ngram in (
    ("word", "word", (1, 2)),
    ("char", "char", (3, 5)),
):
    tr, ytr, te, yte = texts_xy(df, "exact", train_idx, test_idx)
    Xtr, Xte = vectorize(tr, te, analyzer, ngram)
    for depth in (5, 10, 20, None):
        rows.append(fit_rf(Xtr, ytr, Xte, yte, "depth_seed42", feature, depth, 42))
        save(rows)

# Depth sweep, seed 21, word forest only.
train_idx, test_idx = train_test_split(
    df.index, test_size=0.3, stratify=df["y"], random_state=21
)
tr, ytr, te, yte = texts_xy(df, "exact", train_idx, test_idx)
Xtr, Xte = vectorize(tr, te, "word", (1, 2))
for depth in (5, 10, 20, None):
    rows.append(fit_rf(Xtr, ytr, Xte, yte, "depth_seed21", "word", depth, 21))
    save(rows)
print("done", flush=True)
