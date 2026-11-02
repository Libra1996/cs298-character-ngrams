"""Attack-family counts on the issue-3 grouped split."""
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
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))


def preprocess_text(text, stemmer, stop):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


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
df["log_text"] = raw.map(lambda t: preprocess_text(t, stemmer, stop))
dedup = df.drop_duplicates(subset=["log_text", "y"]).copy()
mask = grouped_split(dedup)
test, train = dedup.loc[mask], dedup.loc[~mask]
names = test["attack_name"].to_numpy()
yte = test["y"].to_numpy()


def report(tag, pred):
    for lab in ("SSH-Bruteforce", "FTP-BruteForce"):
        m = names == lab
        caught = int(((pred == 1) & m).sum())
        print(f"{tag} {lab} {caught}/{int(m.sum())}", flush=True)


word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
Xtr_w = word.fit_transform(train["log_text"])
Xte_w = word.transform(test["log_text"])
Xtr_c = char.fit_transform(train["log_text"])
Xte_c = char.transform(test["log_text"])
ytr = train["y"].to_numpy()
for tag, A, B in (("word", Xtr_w, Xte_w), ("char", Xtr_c, Xte_c)):
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(A, ytr)
    report(tag, lr.predict(B))

skip = {"Timestamp", "Label", "y", "log_text", "key", "attack_name"}
cols = [c for c in train.columns if c not in skip]
print("numeric features", len(cols), flush=True)
Xtr = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
Xte = test[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
imp = SimpleImputer(strategy="median")
scaler = StandardScaler()
Xtr = scaler.fit_transform(imp.fit_transform(Xtr))
Xte = scaler.transform(imp.transform(Xte))
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(Xtr, ytr)
report("numeric", rf.predict(Xte))
