"""Grouped split on port bin, protocol, and packet counts, and the both-label strings removed.

The grouped split keeps every (port bin, protocol, forward packets, backward
packets) group entirely in training or entirely in testing. The removal run
drops the 52 strings that carry both labels, then repeats the seed-42 split.
"""
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


def score_text(name, feature, train_text, ytr, test_text, yte, analyzer, ngram):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=10000)
    Xtr = vec.fit_transform(train_text)
    Xte = vec.transform(test_text)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(Xtr, ytr)
    pred = lr.predict(Xte)
    proba = lr.predict_proba(Xte)[:, 1]
    row = {
        "setting": name,
        "feature": feature,
        "model": "Logistic Regression",
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_pos_test": int(np.asarray(yte).sum()),
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
        "auc": float(roc_auc_score(yte, proba)),
        "attack_recall": float(((pred == 1) & (np.asarray(yte) == 1)).sum() / max(np.asarray(yte).sum(), 1)),
    }
    print(
        f"{name} {feature} LR F1 {row['f1_macro']:.4f} Acc {row['accuracy']:.4f} "
        f"AUC {row['auc']:.4f} atkR {row['attack_recall']:.4f} "
        f"pos {row['n_pos_test']}/{row['n_test']}",
        flush=True,
    )
    return row


def score_numeric(name, train, test):
    skip = {"Timestamp", "Label", "y", "log_text", "key", "attack_name"}
    cols = [c for c in train.columns if c not in skip]
    Xtr = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xte = test[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imp.fit_transform(Xtr))
    Xte = scaler.transform(imp.transform(Xte))
    ytr = train["y"].to_numpy()
    yte = test["y"].to_numpy()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    proba = rf.predict_proba(Xte)[:, 1]
    row = {
        "setting": name,
        "feature": "Numeric-78",
        "model": "Random Forest",
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_pos_test": int(yte.sum()),
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
        "auc": float(roc_auc_score(yte, proba)) if yte.sum() not in (0, len(yte)) else 0.5,
        "attack_recall": float(((pred == 1) & (yte == 1)).sum() / max(yte.sum(), 1)),
    }
    print(
        f"{name} numeric RF F1 {row['f1_macro']:.4f} Acc {row['accuracy']:.4f} "
        f"AUC {row['auc']:.4f} atkR {row['attack_recall']:.4f}",
        flush=True,
    )
    return row


def grouped_split(frame, seed=42, test_frac=0.3):
    rng = np.random.RandomState(seed)
    sizes = frame.groupby("key").size()
    has_atk = frame.groupby("key")["y"].sum() > 0
    test_keys = set()
    for flag in (True, False):
        keys = has_atk[has_atk == flag].index.to_numpy().copy()
        rng.shuffle(keys)
        target = test_frac * float(sizes.loc[keys].sum())
        taken = 0.0
        for key in keys:
            if taken >= target and taken > 0:
                break
            test_keys.add(key)
            taken += float(sizes.loc[key])
    return frame["key"].isin(test_keys).to_numpy()


print("Loading...", flush=True)
df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
df = df.drop_duplicates().dropna()
df["attack_name"] = df["Label"]
df["y"] = (df["Label"] != "Benign").astype(int)
dst = np.where(df["Dst Port"].to_numpy() < 1024, "wellknown", "dynamic")
df["key"] = (
    pd.Series(dst, index=df.index)
    + "|"
    + df["Protocol"].astype(str)
    + "|"
    + df["Tot Fwd Pkts"].astype(str)
    + "|"
    + df["Tot Bwd Pkts"].astype(str)
)
raw = (
    "dst_" + pd.Series(dst, index=df.index)
    + " proto_" + df["Protocol"].astype(str)
    + " dur_" + df["Flow Duration"].astype(str)
    + " fwd_" + df["Tot Fwd Pkts"].astype(str)
    + " bwd_" + df["Tot Bwd Pkts"].astype(str)
)
stemmer = PorterStemmer()
stop = set(stopwords.words("english"))
df["log_text"] = raw.map(lambda t: preprocess_text(t, stemmer, stop))
dedup = df.drop_duplicates(subset=["log_text", "y"]).copy()

both = dedup.groupby("log_text")["y"].nunique()
both_strings = set(both[both > 1].index)
print("both_label_strings", len(both_strings), "rows", int(dedup["log_text"].isin(both_strings).sum()), flush=True)

atk = dedup[dedup["y"] == 1]
print("attack rows", len(atk), "attack groups", atk["key"].nunique(), flush=True)
print(atk.groupby("attack_name")["key"].nunique().to_string(), flush=True)
top = atk.groupby(["attack_name", "key"]).size().sort_values(ascending=False).head(8)
print(top.to_string(), flush=True)
print("all groups", dedup["key"].nunique(), "rows", len(dedup), flush=True)

rows = []

# --- Both-label strings removed, same stratified protocol ---
clean = dedup[~dedup["log_text"].isin(both_strings)].copy()
print("after removal", len(clean), "atk", int(clean["y"].sum()), flush=True)
tr_i, te_i = train_test_split(
    clean.index, test_size=0.3, stratify=clean["y"], random_state=42
)
train_c, test_c = clean.loc[tr_i], clean.loc[te_i]
rows.append(score_text("removed_both_labels", "Lexical-Word", train_c["log_text"], train_c["y"], test_c["log_text"], test_c["y"], "word", (1, 2)))
rows.append(score_text("removed_both_labels", "Structural-Char", train_c["log_text"], train_c["y"], test_c["log_text"], test_c["y"], "char", (3, 5)))

# --- Grouped split on the published deduped rows ---
mask = grouped_split(dedup)
test_g, train_g = dedup.loc[mask], dedup.loc[~mask]
print(
    "grouped train", len(train_g), "atk", int(train_g["y"].sum()),
    train_g.loc[train_g["y"] == 1, "attack_name"].value_counts().to_dict(),
    flush=True,
)
print(
    "grouped test", len(test_g), "atk", int(test_g["y"].sum()),
    test_g.loc[test_g["y"] == 1, "attack_name"].value_counts().to_dict(),
    flush=True,
)
print("shared keys", len(set(train_g["key"]) & set(test_g["key"])), flush=True)
print("train groups", train_g["key"].nunique(), "test groups", test_g["key"].nunique(), flush=True)
rows.append(score_text("grouped", "Lexical-Word", train_g["log_text"], train_g["y"], test_g["log_text"], test_g["y"], "word", (1, 2)))
rows.append(score_text("grouped", "Structural-Char", train_g["log_text"], train_g["y"], test_g["log_text"], test_g["y"], "char", (3, 5)))
rows.append(score_numeric("grouped", train_g, test_g))

out = pd.DataFrame(rows)
path = os.path.join(ROOT, "results_issue3_grouped_and_labels.csv")
out.to_csv(path, index=False)
print("wrote", path, flush=True)
