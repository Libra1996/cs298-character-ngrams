"""Per-attack-family counts on the hour split that puts both families on both sides."""
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.abspath(__file__))
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
    return float(tpr[ok[-1]]) if len(ok) else 0.0


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

fam_tr = df[df["hour"].isin(TRAIN_HOURS)].drop_duplicates(subset=["log_text", "y"])
fam_te = df[df["hour"].isin(TEST_HOURS)].drop_duplicates(subset=["log_text", "y"])
ftp_tr = set(fam_tr.loc[fam_tr["Label"] == "FTP-BruteForce", "log_text"])
ftp_te = fam_te.loc[fam_te["Label"] == "FTP-BruteForce", "log_text"]
ssh_tr = set(fam_tr.loc[fam_tr["Label"] == "SSH-Bruteforce", "log_text"])
ssh_te = fam_te.loc[fam_te["Label"] == "SSH-Bruteforce", "log_text"]
print("ftp train strings", len(ftp_tr), "ftp test", len(ftp_te), "overlap", int(ftp_te.isin(ftp_tr).sum()), flush=True)
print("ssh train strings", len(ssh_tr), "ssh test", len(ssh_te), "overlap", int(ssh_te.isin(ssh_tr).sum()), flush=True)

word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
Xtr_w = word.fit_transform(fam_tr["log_text"])
Xte_w = word.transform(fam_te["log_text"])
Xtr_c = char.fit_transform(fam_tr["log_text"])
Xte_c = char.transform(fam_te["log_text"])
ytr = fam_tr["y"].to_numpy()
yte = fam_te["y"].to_numpy()
labels = fam_te["Label"].to_numpy()

from scipy.sparse import csr_matrix, hstack

blocks = {
    "Lexical-Word": (Xtr_w, Xte_w),
    "Structural-Char": (Xtr_c, Xte_c),
    "Semantic-Entity": (csr_matrix((len(fam_tr), 5)), csr_matrix((len(fam_te), 5))),
    "Hybrid-Robust": (hstack([Xtr_c, csr_matrix((len(fam_tr), 5))]), hstack([Xte_c, csr_matrix((len(fam_te), 5))])),
}
models = []
for config, (A, B) in blocks.items():
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    lr.fit(A, ytr)
    rf.fit(A, ytr)
    p_lr = lr.predict_proba(B)[:, 1]
    p_rf = rf.predict_proba(B)[:, 1]
    models.append((config, "Logistic Regression", p_lr))
    models.append((config, "Random Forest", p_rf))
    models.append((config, "Soft-Voting Ensemble", (p_lr + p_rf) / 2.0))

rows = []
for config, name, proba in models:
    pred = (proba >= 0.5).astype(int)
    for lab in ("SSH-Bruteforce", "FTP-BruteForce"):
        mask = labels == lab
        n = int(mask.sum())
        caught = int(((pred == 1) & mask).sum())
        # recall at 1% FPR uses the full test set's benign rows, then counts this family
        fpr, _, thr = roc_curve(yte, proba)
        ok = np.where(fpr <= 0.01 + 1e-12)[0]
        cut = thr[ok[-1]] if len(ok) else np.inf
        caught_1 = int(((proba >= cut) & mask).sum())
        print(
            f"{config} {name} {lab} {caught}/{n} at 0.5; {caught_1}/{n} at 1% FPR",
            flush=True,
        )
        rows.append(
            {
                "config": config,
                "model": name,
                "label": lab,
                "n": n,
                "caught_at_0.5": caught,
                "caught_at_fpr_0.01": caught_1,
            }
        )

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results_issue2_family_by_label.csv"), index=False)
print("wrote results_issue2_family_by_label.csv", flush=True)
