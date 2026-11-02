"""Per-class precision, recall, false-positive rate, and precision-recall AUC.

Seed-42 IDS2018 ablation, seed-42 CTI ablation, and the 10:00 time split.
"""
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
IOC = {
    "IPV4", "URL", "DOMAIN", "MD5", "SHA1", "SHA2", "FILEPATH", "REGISTRYKEY",
    "EMAIL", "malware", "vulnerability", "threat-actor", "attack-pattern",
    "campaign", "tools",
}
ENTITY_PATTERNS = {
    "count_ip": re.compile(
        r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?!\d)"
    ),
    "count_url": re.compile(
        r"(?:https?://|www\.)[^\s]+|(?:[a-z0-9-]+\.)+(?:com|net|org|io|edu|gov)(?:/[^\s]*)?",
        re.IGNORECASE,
    ),
    "count_cve": re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE),
    "count_hash": re.compile(r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b"),
    "count_path": re.compile(
        r"(?:[A-Za-z]:\\(?:[^\s\\/:*?\"<>|\r\n]+\\)*[^\s\\/:*?\"<>|\r\n]+)"
        r"|(?:/(?:(?:etc|var|usr|home|tmp|opt|bin|sbin)/|\S+/)+)\S+"
    ),
}
FEATURE_COLS = list(ENTITY_PATTERNS.keys())


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


def class_rows(y_true, y_pred, proba, dataset, config, model, names=("Benign", "Attack")):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = []
    for cls, name in ((0, names[0]), (1, names[1])):
        tp = int(cm[cls, cls])
        fp = int(cm[:, cls].sum() - tp)
        fn = int(cm[cls, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        scores = proba[:, cls]
        pr_auc = average_precision_score((y_true == cls).astype(int), scores)
        out.append(
            {
                "Dataset": dataset,
                "Feature Config": config,
                "Model": model,
                "Class": name,
                "Precision": precision,
                "Recall": recall,
                "FPR": fpr,
                "PR-AUC": pr_auc,
                "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
        )
    return out


def score_configs(configs, ytr, yte, dataset, rows, names=("Benign", "Attack")):
    for config, (Xtr, Xte) in configs.items():
        for name, model in make_models().items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            proba = model.predict_proba(Xte)
            # Column order follows model.classes_
            ordered = np.zeros_like(proba)
            for i, cls in enumerate(model.classes_):
                ordered[:, int(cls)] = proba[:, i]
            got = class_rows(yte, pred, ordered, dataset, config, name, names)
            rows.extend(got)
            print(
                f"  {dataset} {config} {name} F1={got[0]['F1_macro']:.4f} "
                f"atk P={got[1]['Precision']:.4f} R={got[1]['Recall']:.4f} "
                f"FPR={got[1]['FPR']:.4f} PR={got[1]['PR-AUC']:.4f}",
                flush=True,
            )


def preprocess_factory():
    stemmer = PorterStemmer()
    stop = set(stopwords.words("english"))

    def preprocess_text(text):
        tokens = re.findall(r"\b\w+\b", str(text).lower())
        return " ".join(stemmer.stem(w) for w in tokens if w not in stop)

    return preprocess_text


def entity_matrix(texts):
    rows = []
    for text in texts:
        rows.append({name: len(pat.findall(text)) for name, pat in ENTITY_PATTERNS.items()})
    return pd.DataFrame(rows)[FEATURE_COLS]


print("Loading IDS2018...", flush=True)
df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
df = df.drop_duplicates().dropna()
df["ts"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
df["y"] = (df["Label"] != "Benign").astype(int)
dst = np.where(df["Dst Port"].to_numpy() < 1024, "wellknown", "dynamic")
raw = (
    "dst_" + pd.Series(dst, index=df.index)
    + " proto_" + df["Protocol"].astype(str)
    + " dur_" + df["Flow Duration"].astype(str)
    + " fwd_" + df["Tot Fwd Pkts"].astype(str)
    + " bwd_" + df["Tot Bwd Pkts"].astype(str)
)
prep = preprocess_factory()
df["log_text"] = raw.map(prep)
rows = []

dedup = df.drop_duplicates(subset=["log_text", "y"])
X_train, X_test, y_train, y_test = train_test_split(
    dedup["log_text"], dedup["y"], test_size=0.3, stratify=dedup["y"], random_state=42
)
Xtr, Xte = X_train.tolist(), X_test.tolist()
ytr, yte = y_train.to_numpy(), y_test.to_numpy()
word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
Xtr_w, Xte_w = word.fit_transform(Xtr), word.transform(Xte)
Xtr_c, Xte_c = char.fit_transform(Xtr), char.transform(Xte)
ztr, zte = csr_matrix((len(Xtr), 5)), csr_matrix((len(Xte), 5))
print("IDS2018 seed 42", flush=True)
score_configs(
    {
        "Lexical-Word": (Xtr_w, Xte_w),
        "Structural-Char": (Xtr_c, Xte_c),
        "Semantic-Entity": (ztr, zte),
        "Hybrid-Robust": (hstack([Xtr_c, ztr]), hstack([Xte_c, zte])),
    },
    ytr,
    yte,
    "IDS2018",
    rows,
)

print("CTI seed 42", flush=True)
cti = pd.read_csv(os.path.join(ROOT, "data", "cti", "cyber_threat_intelligence.csv"))
cti["text"] = cti["text"].fillna("").astype(str)


def gold(entity_labels):
    if entity_labels is None or (isinstance(entity_labels, float) and np.isnan(entity_labels)):
        return 0
    labels = set(str(entity_labels).split("|")) - {""}
    return int(len(labels & IOC) > 0)


cti["y"] = cti["entity_labels"].apply(gold)
Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(
    cti["text"], cti["y"], test_size=0.3, stratify=cti["y"], random_state=42
)
Xtr_t, Xte_t = Xtr_t.tolist(), Xte_t.tolist()
ytr_t, yte_t = ytr_t.to_numpy(), yte_t.to_numpy()
word_c = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
char_c = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
Xtr_wc, Xte_wc = word_c.fit_transform(Xtr_t), word_c.transform(Xte_t)
Xtr_cc, Xte_cc = char_c.fit_transform(Xtr_t), char_c.transform(Xte_t)
ent_tr = entity_matrix(Xtr_t)
ent_te = entity_matrix(Xte_t)
scaler = StandardScaler(with_mean=True)
ent_tr_s = csr_matrix(scaler.fit_transform(ent_tr.astype(float)))
ent_te_s = csr_matrix(scaler.transform(ent_te.astype(float)))
score_configs(
    {
        "Lexical-Word": (Xtr_wc, Xte_wc),
        "Structural-Char": (Xtr_cc, Xte_cc),
        "Semantic-Entity": (ent_tr_s, ent_te_s),
        "Hybrid-Robust": (hstack([Xtr_cc, ent_tr_s]), hstack([Xte_cc, ent_te_s])),
    },
    ytr_t,
    yte_t,
    "CTI",
    rows,
    names=("Negative", "Positive"),
)

print("Time split", flush=True)
kept = df[df["ts"] >= "2018-01-01"].copy()
train = kept[kept["ts"] < "2018-02-14 10:00:00"].drop_duplicates(subset=["log_text", "y"])
test = kept[kept["ts"] >= "2018-02-14 10:00:00"].drop_duplicates(subset=["log_text", "y"])
Xtr, Xte = train["log_text"].tolist(), test["log_text"].tolist()
ytr, yte = train["y"].to_numpy(), test["y"].to_numpy()
word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
Xtr_w, Xte_w = word.fit_transform(Xtr), word.transform(Xte)
Xtr_c, Xte_c = char.fit_transform(Xtr), char.transform(Xte)
ztr, zte = csr_matrix((len(Xtr), 5)), csr_matrix((len(Xte), 5))
score_configs(
    {
        "Lexical-Word": (Xtr_w, Xte_w),
        "Structural-Char": (Xtr_c, Xte_c),
        "Semantic-Entity": (ztr, zte),
        "Hybrid-Robust": (hstack([Xtr_c, ztr]), hstack([Xte_c, zte])),
    },
    ytr,
    yte,
    "IDS2018-time",
    rows,
)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results_class_metrics.csv"), index=False)
print("wrote results_class_metrics.csv", flush=True)
