"""Grouped split for seeds 42, 7, 13, 21, and 99. Forest seed stays 42."""
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 7, 13, 21, 99]
OUT = os.path.join(ROOT, "results_grouped_seeds.csv")


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


def grouped_split(frame, seed, test_frac=0.3):
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


def metrics(name, feature, model, yte, pred, proba, names, n_train, seed, ftp_in_test):
    yte = np.asarray(yte)
    pred = np.asarray(pred)
    tp = int(((pred == 1) & (yte == 1)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    ssh = names == "SSH-Bruteforce"
    ftp = names == "FTP-BruteForce"
    row = {
        "seed": seed,
        "feature": feature,
        "model": model,
        "n_train": int(n_train),
        "n_test": int(len(yte)),
        "n_pos_test": int(yte.sum()),
        "attack_share": float(yte.mean()),
        "ftp_in_test": int(ftp_in_test),
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
        "auc": float(roc_auc_score(yte, proba)) if yte.sum() not in (0, len(yte)) else 0.5,
        "attack_precision": float(precision_score(yte, pred, zero_division=0)),
        "attack_recall": float(recall_score(yte, pred, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "ssh_n": int(ssh.sum()),
        "ssh_caught": int(((pred == 1) & ssh).sum()),
        "ftp_n": int(ftp.sum()),
        "ftp_caught": int(((pred == 1) & ftp).sum()),
    }
    print(
        f"seed {seed} {name} F1 {row['f1_macro']:.4f} "
        f"P {row['attack_precision']:.4f} R {row['attack_recall']:.4f} "
        f"share {row['attack_share']:.4f} FTP {row['ftp_caught']}/{row['ftp_n']}",
        flush=True,
    )
    return row


def score_text(feature, analyzer, ngram, train, test, seed, ftp_in_test):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=10000)
    Xtr = vec.fit_transform(train["log_text"])
    Xte = vec.transform(test["log_text"])
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(Xtr, train["y"].to_numpy())
    pred = lr.predict(Xte)
    proba = lr.predict_proba(Xte)[:, 1]
    return metrics(
        feature, feature, "Logistic Regression",
        test["y"], pred, proba, test["attack_name"].to_numpy(),
        len(train), seed, ftp_in_test,
    )


def score_numeric(train, test, seed, ftp_in_test):
    drop = {"Timestamp", "Label", "y", "log_text", "key", "attack_name"}
    cols = [c for c in train.columns if c not in drop]
    Xtr = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xte = test[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imp.fit_transform(Xtr))
    Xte = scaler.transform(imp.transform(Xte))
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(Xtr, train["y"].to_numpy())
    pred = rf.predict(Xte)
    proba = rf.predict_proba(Xte)[:, 1]
    return metrics(
        "Numeric-78", "Numeric-78", "Random Forest",
        test["y"], pred, proba, test["attack_name"].to_numpy(),
        len(train), seed, ftp_in_test,
    )


def load_dedup():
    cache = os.path.join(ROOT, "cache_dedup_grouped.pkl")
    if os.path.exists(cache):
        print("Loading cache", flush=True)
        return pd.read_pickle(cache)
    print("Loading CSV...", flush=True)
    df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
    df = df.drop_duplicates().dropna()
    df["attack_name"] = df["Label"]
    df["y"] = (df["Label"] != "Benign").astype(int)
    dst = np.where(df["Dst Port"].to_numpy() < 1024, "wellknown", "dynamic")
    df["key"] = (
        pd.Series(dst, index=df.index)
        + "|" + df["Protocol"].astype(str)
        + "|" + df["Tot Fwd Pkts"].astype(str)
        + "|" + df["Tot Bwd Pkts"].astype(str)
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
    print("Stemming...", flush=True)
    df["log_text"] = raw.map(lambda t: preprocess_text(t, stemmer, stop))
    dedup = df.drop_duplicates(subset=["log_text", "y"]).copy()
    dedup.to_pickle(cache)
    print("cached", len(dedup), flush=True)
    return dedup


def main():
    dedup = load_dedup()
    rows = []
    for seed in SEEDS:
        mask = grouped_split(dedup, seed)
        test = dedup.loc[mask]
        train = dedup.loc[~mask]
        ftp_in_test = bool((test["attack_name"] == "FTP-BruteForce").any())
        print(
            f"seed {seed} train {len(train)} test {len(test)} "
            f"atk {int(test['y'].sum())} share {test['y'].mean():.4f} "
            f"ftp_in_test {ftp_in_test}",
            flush=True,
        )
        rows.append(score_text("Lexical-Word", "word", (1, 2), train, test, seed, ftp_in_test))
        rows.append(score_text("Structural-Char", "char", (3, 5), train, test, seed, ftp_in_test))
        rows.append(score_numeric(train, test, seed, ftp_in_test))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
