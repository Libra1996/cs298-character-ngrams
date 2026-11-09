"""Binned duration on the 10:00 cut and the hour split."""
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_binned_time_splits.csv")
CUT = pd.Timestamp("2018-02-14 10:00:00")
TRAIN_HOURS = {1, 2, 10}
TEST_HOURS = {3, 4, 5, 8, 9, 11, 12}


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


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


def load_frame():
    cache = os.path.join(ROOT, "cache_binned_flows.pkl")
    if os.path.exists(cache):
        print("Loading binned cache", flush=True)
        return pd.read_pickle(cache)
    print("Loading CSV...", flush=True)
    df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
    df = df.drop_duplicates().dropna()
    df["ts"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
    df = df[df["ts"] >= "2018-01-01"].copy()
    df["y"] = (df["Label"] != "Benign").astype(int)
    df["hour"] = df["ts"].dt.hour
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
    print("Stemming...", flush=True)
    exact = raw.map(lambda t: preprocess_text(t, stemmer, stop))
    dur = df["Flow Duration"].astype("int64")
    binned = [
        re.sub(r"dur_\d+", duration_bin(d), text)
        for text, d in zip(exact, dur)
    ]
    out = pd.DataFrame(
        {
            "ts": df["ts"].to_numpy(),
            "hour": df["hour"].to_numpy(),
            "y": df["y"].to_numpy(),
            "label": df["Label"].to_numpy(),
            "text": binned,
        }
    )
    out.to_pickle(cache)
    print("cached", len(out), flush=True)
    return out


def pack(y, pred, labels):
    y = np.asarray(y)
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    ftp = labels == "FTP-BruteForce"
    ssh = labels == "SSH-Bruteforce"
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "attack_precision": float(precision_score(y, pred, zero_division=0)),
        "attack_recall": float(recall_score(y, pred, zero_division=0)),
        "ftp_n": int(ftp.sum()),
        "ftp_caught": int(((pred == 1) & ftp).sum()),
        "ssh_n": int(ssh.sum()),
        "ssh_caught": int(((pred == 1) & ssh).sum()),
    }


def run_split(name, train_flows, test_flows):
    train = train_flows.drop_duplicates(subset=["text", "y"])
    test_strings = test_flows.drop_duplicates(subset=["text", "y"])
    print(
        f"{name} train strings {len(train)} test strings {len(test_strings)} "
        f"test flows {len(test_flows)}",
        flush=True,
    )
    rows = []
    specs = {
        "Lexical-Word": ("word", (1, 2)),
        "Structural-Char": ("char", (3, 5)),
    }
    for feature, (analyzer, ngram) in specs.items():
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=10000)
        Xtr = vec.fit_transform(train["text"])
        Xte_s = vec.transform(test_strings["text"])
        Xte_f = vec.transform(test_flows["text"])
        ytr = train["y"].to_numpy()
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
        )
        lr.fit(Xtr, ytr)
        rf.fit(Xtr, ytr)
        fitted = {
            "Logistic Regression": (lr.predict(Xte_s), lr.predict(Xte_f)),
            "Random Forest": (rf.predict(Xte_s), rf.predict(Xte_f)),
        }
        p_lr_s = lr.predict_proba(Xte_s)[:, 1]
        p_rf_s = rf.predict_proba(Xte_s)[:, 1]
        p_lr_f = lr.predict_proba(Xte_f)[:, 1]
        p_rf_f = rf.predict_proba(Xte_f)[:, 1]
        fitted["Soft-Voting Ensemble"] = (
            ((p_lr_s + p_rf_s) / 2.0 >= 0.5).astype(int),
            ((p_lr_f + p_rf_f) / 2.0 >= 0.5).astype(int),
        )
        for model, (pred_s, pred_f) in fitted.items():
            for unit, pred, frame in (
                ("string", pred_s, test_strings),
                ("flow", pred_f, test_flows),
            ):
                row = pack(frame["y"], pred, frame["label"])
                row.update({"split": name, "unit": unit, "feature": feature, "model": model})
                rows.append(row)
                print(
                    f"{name} {unit} {feature} {model} F1 {row['f1_macro']:.4f} "
                    f"P {row['attack_precision']:.4f} R {row['attack_recall']:.4f} "
                    f"FTP {row['ftp_caught']}/{row['ftp_n']}",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(OUT, index=False)
    return rows


def main():
    df = load_frame()
    print("hour counts", df.groupby("hour").size().to_dict(), flush=True)
    cut_tr = df[df["ts"] < CUT]
    cut_te = df[df["ts"] >= CUT]
    ftp = cut_te[cut_te["label"] == "FTP-BruteForce"]
    print(
        "10:00 FTP flows", len(ftp),
        "unique binned strings", ftp["text"].nunique(),
        flush=True,
    )
    rows = []
    rows.extend(run_split("10:00", cut_tr, cut_te))
    hour_tr = df[df["hour"].isin(TRAIN_HOURS)]
    hour_te = df[df["hour"].isin(TEST_HOURS)]
    rows.extend(run_split("hour", hour_tr, hour_te))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
