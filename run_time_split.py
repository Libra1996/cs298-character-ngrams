"""Train on flows before 10:00 and test on flows from 10:00 on."""
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
CUT = pd.Timestamp("2018-02-14 10:00:00")


def make_models():
    return {
        "Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
        ),
        "Soft-Voting Ensemble": VotingClassifier(
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
        ),
    }


print("Loading...", flush=True)
df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
df = df.drop_duplicates().dropna()
df["ts"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
n_bad = int((df["ts"] < "2018-01-01").sum())
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


def preprocess_text(text):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


df["log_text"] = raw.map(preprocess_text)
train = df[df["ts"] < CUT]
test = df[df["ts"] >= CUT]
print(f"bad_timestamps_dropped {n_bad}", flush=True)
print(
    f"rows before text dedup train {len(train)} attack {int(train['y'].sum())} "
    f"test {len(test)} attack {int(test['y'].sum())}",
    flush=True,
)
print(f"train time {train['ts'].min()} .. {train['ts'].max()}", flush=True)
print(f"test time {test['ts'].min()} .. {test['ts'].max()}", flush=True)

train = train.drop_duplicates(subset=["log_text", "y"])
test = test.drop_duplicates(subset=["log_text", "y"])
overlap = len(set(train["log_text"]) & set(test["log_text"]))
print(
    f"after text dedup train {len(train)} attack {int(train['y'].sum())} "
    f"test {len(test)} attack {int(test['y'].sum())} "
    f"shared_strings {overlap}",
    flush=True,
)

Xtr = train["log_text"].tolist()
Xte = test["log_text"].tolist()
ytr = train["y"].to_numpy()
yte = test["y"].to_numpy()
word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
Xtr_w, Xte_w = word.fit_transform(Xtr), word.transform(Xte)
Xtr_c, Xte_c = char.fit_transform(Xtr), char.transform(Xte)
ztr, zte = csr_matrix((len(Xtr), 5)), csr_matrix((len(Xte), 5))
configs = {
    "Lexical-Word": (Xtr_w, Xte_w),
    "Structural-Char": (Xtr_c, Xte_c),
    "Semantic-Entity": (ztr, zte),
    "Hybrid-Robust": (hstack([Xtr_c, ztr]), hstack([Xte_c, zte])),
}
rows = []
for config, (A, B) in configs.items():
    for name, model in make_models().items():
        model.fit(A, ytr)
        pred = model.predict(B)
        proba = model.predict_proba(B)[:, 1]
        row = {
            "Feature Config": config,
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "F1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "AUC": roc_auc_score(yte, proba),
        }
        rows.append(row)
        print(
            f"{config} {name} Acc={row['Accuracy']:.4f} "
            f"F1={row['F1_macro']:.4f} AUC={row['AUC']:.4f}",
            flush=True,
        )

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results_time_split.csv"), index=False)
print("wrote results_time_split.csv", flush=True)
