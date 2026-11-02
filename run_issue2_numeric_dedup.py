"""Numeric models on the 10:00 cut after (text, label) dedup.

Same surviving rows the text models see: one row per string, so the
53 FTP strings can be compared directly with the text counts.
"""
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
CUT = pd.Timestamp("2018-02-14 10:00:00")


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


def recall_at_fpr(y, proba, max_fpr=0.01):
    y = np.asarray(y)
    fpr, tpr, _ = roc_curve(y, proba)
    ok = np.where(fpr <= max_fpr + 1e-12)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


print("Loading...", flush=True)
df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
df = df.drop_duplicates().dropna()
df["ts"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
df = df[df["ts"] >= "2018-01-01"].copy()
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
df["log_text"] = raw.map(lambda t: preprocess_text(t, stemmer, stop))

train = df[df["ts"] < CUT].drop_duplicates(subset=["log_text", "y"])
test = df[df["ts"] >= CUT].drop_duplicates(subset=["log_text", "y"])
print(
    "dedup train", len(train), "atk", int(train["y"].sum()),
    "test", len(test), "atk", int(test["y"].sum()),
    flush=True,
)

skip = {"Timestamp", "Label", "y", "log_text", "ts"}
num_cols = [c for c in df.columns if c not in skip]
Xtr = train[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
Xte = test[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
imp = SimpleImputer(strategy="median")
scaler = StandardScaler()
Xtr = scaler.fit_transform(imp.fit_transform(Xtr))
Xte = scaler.transform(imp.transform(Xte))
ytr = train["y"].to_numpy()
yte = test["y"].to_numpy()
print("features", Xtr.shape[1], flush=True)

lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
lr.fit(Xtr, ytr)
rf.fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]
p_rf = rf.predict_proba(Xte)[:, 1]
probas = {
    "Logistic Regression": p_lr,
    "Random Forest": p_rf,
    "Soft-Voting Ensemble": (p_lr + p_rf) / 2.0,
}
rows = []
n_pos = int(yte.sum())
for name, proba in probas.items():
    caught = int((((proba >= 0.5).astype(int) == 1) & (yte == 1)).sum())
    rec1 = recall_at_fpr(yte, proba, 0.01)
    auc = float(roc_auc_score(yte, proba))
    print(f"{name} caught {caught}/{n_pos} rec1% {rec1:.4f} auc {auc:.6f}", flush=True)
    rows.append(
        {
            "split": "forward10_numeric_text_dedup",
            "model": name,
            "n_pos": n_pos,
            "n_test": int(len(yte)),
            "caught_at_0.5": caught,
            "recall_at_fpr_0.01": rec1,
            "auc": auc,
        }
    )

out = os.path.join(ROOT, "results_issue2_numeric_dedup.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out, flush=True)
