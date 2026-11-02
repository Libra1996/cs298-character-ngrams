"""Random forest and logistic regression on the 78 numeric flow columns. Same rows and split."""
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
df = df.drop_duplicates().dropna()
df["y"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)

dst = np.where(df["Dst Port"] < 1024, "wellknown", "dynamic")
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
df = df.drop_duplicates(subset=["log_text", "y"])

num_cols = [c for c in df.columns if c not in ("Timestamp", "Label", "y", "log_text")]
X_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
X_num = X_num.replace([np.inf, -np.inf], np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X_num,
    df["y"],
    test_size=0.3,
    stratify=df["y"],
    random_state=42,
)

imp = SimpleImputer(strategy="median")
scaler = StandardScaler()
Xtr = scaler.fit_transform(imp.fit_transform(X_train))
Xte = scaler.transform(imp.transform(X_test))
print("numeric_features", Xtr.shape[1], "train", Xtr.shape[0], "test", Xte.shape[0])

lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
)
ens = VotingClassifier(
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

rows = []
for name, model in (
    ("Logistic Regression", lr),
    ("Random Forest", rf),
    ("Soft-Voting Ensemble", ens),
):
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "F1_macro": f1_score(y_test, pred, average="macro"),
        "AUC": roc_auc_score(y_test, proba),
    }
    rows.append(row)
    print(row)

out = pd.DataFrame(rows)
out_path = os.path.join(ROOT, "results_numeric_baseline.csv")
out.to_csv(out_path, index=False)
print("wrote", out_path)
