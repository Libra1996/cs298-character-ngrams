"""February 14 clock cuts.

Forward 10:00 cut: text models after (text, label) dedup, plus numeric
models on the same clock cut before that dedup.
Family-hour cut: earlier hour of each attack period in training and the
later hours in testing, so SSH-Bruteforce and FTP-BruteForce are on both sides.
"""
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
CUT = pd.Timestamp("2018-02-14 10:00:00")
TRAIN_HOURS = {1, 2, 10}
TEST_HOURS = {3, 4, 5, 8, 9, 11, 12}


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


def recall_at_fpr(y, proba, max_fpr=0.01):
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.0
    fpr, tpr, _ = roc_curve(y, proba)
    ok = np.where(fpr <= max_fpr + 1e-12)[0]
    if len(ok) == 0:
        return 0.0
    return float(tpr[ok[-1]])


def fit_pair(Xtr, ytr, Xte):
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
    )
    lr.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    p_rf = rf.predict_proba(Xte)[:, 1]
    return {
        "Logistic Regression": p_lr,
        "Random Forest": p_rf,
        "Soft-Voting Ensemble": (p_lr + p_rf) / 2.0,
    }


def score_rows(split, config, y, probas):
    y = np.asarray(y)
    n_pos = int(y.sum())
    rows = []
    for name, proba in probas.items():
        pred = (proba >= 0.5).astype(int)
        caught = int(((pred == 1) & (y == 1)).sum())
        rows.append(
            {
                "split": split,
                "config": config,
                "model": name,
                "n_pos": n_pos,
                "n_test": int(len(y)),
                "caught_at_0.5": caught,
                "recall_at_0.5": caught / n_pos if n_pos else 0.0,
                "recall_at_fpr_0.01": recall_at_fpr(y, proba, 0.01),
                "auc": float(roc_auc_score(y, proba)) if n_pos and n_pos < len(y) else 0.5,
            }
        )
        print(
            f"{split} {config} {name} caught {caught}/{n_pos} "
            f"rec05 {rows[-1]['recall_at_0.5']:.4f} "
            f"rec1% {rows[-1]['recall_at_fpr_0.01']:.4f} "
            f"auc {rows[-1]['auc']:.4f}",
            flush=True,
        )
    return rows


def text_matrices(train, test):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
    Xtr_w = word.fit_transform(train["log_text"])
    Xte_w = word.transform(test["log_text"])
    Xtr_c = char.fit_transform(train["log_text"])
    Xte_c = char.transform(test["log_text"])
    ztr = csr_matrix((len(train), 5))
    zte = csr_matrix((len(test), 5))
    return {
        "Lexical-Word": (Xtr_w, Xte_w),
        "Structural-Char": (Xtr_c, Xte_c),
        "Semantic-Entity": (ztr, zte),
        "Hybrid-Robust": (hstack([Xtr_c, ztr]), hstack([Xte_c, zte])),
    }


def numeric_xy(frame):
    num_cols = [c for c in frame.columns if c not in ("Timestamp", "Label", "y", "log_text", "ts", "hour")]
    X = frame[num_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, frame["y"].to_numpy()


def fit_numeric(train, test):
    Xtr_raw, ytr = numeric_xy(train)
    Xte_raw, yte = numeric_xy(test)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imp.fit_transform(Xtr_raw))
    Xte = scaler.transform(imp.transform(Xte_raw))
    print(f"numeric features {Xtr.shape[1]} train {Xtr.shape[0]} test {Xte.shape[0]}", flush=True)
    return yte, fit_pair(Xtr, ytr, Xte)


print("Loading...", flush=True)
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
df["log_text"] = raw.map(lambda t: preprocess_text(t, stemmer, stop))

rows = []

# --- Forward 10:00, numeric on full flows (before text dedup) ---
fwd_tr = df[df["ts"] < CUT]
fwd_te = df[df["ts"] >= CUT]
print(
    "forward raw train", len(fwd_tr), "atk", int(fwd_tr["y"].sum()),
    "labels", fwd_tr.loc[fwd_tr["y"] == 1, "Label"].value_counts().to_dict(),
    flush=True,
)
print(
    "forward raw test", len(fwd_te), "atk", int(fwd_te["y"].sum()),
    "labels", fwd_te.loc[fwd_te["y"] == 1, "Label"].value_counts().to_dict(),
    flush=True,
)
yte, probas = fit_numeric(fwd_tr, fwd_te)
rows.extend(score_rows("forward10_numeric_raw", "Numeric-78", yte, probas))

# --- Forward 10:00, text after within-fold (text, label) dedup ---
fwd_tr_t = fwd_tr.drop_duplicates(subset=["log_text", "y"])
fwd_te_t = fwd_te.drop_duplicates(subset=["log_text", "y"])
print(
    "forward text train", len(fwd_tr_t), "atk", int(fwd_tr_t["y"].sum()),
    "test", len(fwd_te_t), "atk", int(fwd_te_t["y"].sum()),
    flush=True,
)
mats = text_matrices(fwd_tr_t, fwd_te_t)
yte = fwd_te_t["y"].to_numpy()
for config, (A, B) in mats.items():
    rows.extend(score_rows("forward10_text", config, yte, fit_pair(A, fwd_tr_t["y"].to_numpy(), B)))

# --- Family-hour cut: both attack labels on both sides ---
fam_tr = df[df["hour"].isin(TRAIN_HOURS)]
fam_te = df[df["hour"].isin(TEST_HOURS)]
print(
    "family raw train", len(fam_tr),
    fam_tr.loc[fam_tr["y"] == 1, "Label"].value_counts().to_dict(),
    flush=True,
)
print(
    "family raw test", len(fam_te),
    fam_te.loc[fam_te["y"] == 1, "Label"].value_counts().to_dict(),
    flush=True,
)
yte, probas = fit_numeric(fam_tr, fam_te)
rows.extend(score_rows("family_hour_numeric_raw", "Numeric-78", yte, probas))
fam_tr_t = fam_tr.drop_duplicates(subset=["log_text", "y"])
fam_te_t = fam_te.drop_duplicates(subset=["log_text", "y"])
print(
    "family text train", len(fam_tr_t), "atk", int(fam_tr_t["y"].sum()),
    fam_tr_t.loc[fam_tr_t["y"] == 1, "Label"].value_counts().to_dict(),
    flush=True,
)
print(
    "family text test", len(fam_te_t), "atk", int(fam_te_t["y"].sum()),
    fam_te_t.loc[fam_te_t["y"] == 1, "Label"].value_counts().to_dict(),
    flush=True,
)
mats = text_matrices(fam_tr_t, fam_te_t)
yte = fam_te_t["y"].to_numpy()
for config, (A, B) in mats.items():
    rows.extend(score_rows("family_hour_text", config, yte, fit_pair(A, fam_tr_t["y"].to_numpy(), B)))

out = pd.DataFrame(rows)
path = os.path.join(ROOT, "results_issue2_family_split.csv")
out.to_csv(path, index=False)
print("wrote", path, flush=True)
