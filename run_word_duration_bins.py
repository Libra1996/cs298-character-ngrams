"""Word TF-IDF after the duration token is replaced with a bin. Same rows and split."""
import os
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
df = df.drop_duplicates().dropna()
df["y"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)

dst = df["Dst Port"].lt(1024).map({True: "wellknown", False: "dynamic"})
raw = (
    "dst_" + dst.astype(str)
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
df["dur"] = df["Flow Duration"].astype("int64")
df = df.drop_duplicates(subset=["log_text", "y"])


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


df["binned"] = df["log_text"].str.replace(r"dur_\d+", "DURBIN", regex=True)
df["binned"] = [
    text.replace("DURBIN", duration_bin(d))
    for text, d in zip(df["binned"], df["dur"])
]

print("bin_counts")
print(df["binned"].str.extract(r"(dur_\w+)")[0].value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    df["binned"],
    df["y"],
    test_size=0.3,
    stratify=df["y"],
    random_state=42,
)
# The split above is on binned text, which can merge rows that differed only by
# exact duration. For a paired comparison, split the original strings instead.
X_train, X_test, y_train, y_test = train_test_split(
    df["log_text"],
    df["y"],
    test_size=0.3,
    stratify=df["y"],
    random_state=42,
)
train_text = df.loc[X_train.index, "binned"]
test_text = df.loc[X_test.index, "binned"]

vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
Xtr = vec.fit_transform(train_text)
Xte = vec.transform(test_text)
print("vocab", len(vec.vocabulary_), "train", Xtr.shape[0], "test", Xte.shape[0])

models = {
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

rows = []
for name, model in models.items():
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

out = os.path.join(ROOT, "results_word_duration_bins.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
