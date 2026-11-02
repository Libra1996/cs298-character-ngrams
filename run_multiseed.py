"""Repeat the flow ablation, the snippet ablation, and the noise test over five split seeds."""
import os
import random
import re
import string
import time

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 13, 21, 99)
NOISE_RATES = (0.0, 0.05, 0.10, 0.20)
ALPHANUM = string.ascii_lowercase + string.digits
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


def make_models(seed):
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=seed
    )
    ens = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, n_jobs=-1, random_state=seed
                ),
            ),
        ],
        voting="soft",
    )
    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Soft-Voting Ensemble": ens,
    }


def apply_operational_noise(text, rate, rng):
    if rate <= 0 or not text:
        return text
    out = []
    for ch in text:
        if rng.random() < rate:
            op = rng.choice(("sub", "del", "ins"))
            if op == "sub":
                out.append(rng.choice(ALPHANUM))
            elif op == "del":
                continue
            else:
                out.append(ch)
                out.append(rng.choice(ALPHANUM))
        else:
            out.append(ch)
    s = "".join(out)
    delim_out = []
    for ch in s:
        if ch in (" ", "_") and rng.random() < rate:
            delim_out.append(rng.choice((" ", "  ", "_", "-")))
        else:
            delim_out.append(ch)
    s = "".join(delim_out)
    frag = []
    for i, ch in enumerate(s):
        frag.append(ch)
        if ch.isalnum() and i + 1 < len(s) and s[i + 1].isalnum() and rng.random() < rate:
            frag.append(" ")
    return "".join(frag)


def entity_matrix(texts):
    rows = []
    for text in texts:
        rows.append({name: len(pat.findall(text)) for name, pat in ENTITY_PATTERNS.items()})
    frame = pd.DataFrame(rows)[FEATURE_COLS]
    return frame


def score(y_true, model, X):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "AUC": roc_auc_score(y_true, proba),
    }


def fit_and_score(Xtr, Xte, ytr, yte, seed, dataset, config, rows, noise_rate=0.0):
    for name, model in make_models(seed).items():
        model.fit(Xtr, ytr)
        row = score(yte, model, Xte)
        row.update(
            {
                "Seed": seed,
                "Dataset": dataset,
                "Feature Config": config,
                "Model": name,
                "Noise Rate": noise_rate,
            }
        )
        rows.append(row)
        print(
            f"    {dataset} seed={seed} {config} {name} "
            f"rate={noise_rate:.2f} F1={row['F1_macro']:.4f}",
            flush=True,
        )
    return rows[-3]["Model"]  # touch last models via rows only


def train_keep(Xtr, ytr, seed):
    trained = {}
    for name, model in make_models(seed).items():
        model.fit(Xtr, ytr)
        trained[name] = model
    return trained


print("Loading IDS2018...", flush=True)
df = pd.read_csv(os.path.join(ROOT, "02-14-2018.csv"))
df = df.drop_duplicates().dropna()
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
df = df.drop_duplicates(subset=["log_text", "y"])
print(f"IDS2018 rows {len(df)}", flush=True)

cti_path = os.path.join(ROOT, "data", "cti", "cyber_threat_intelligence.csv")
cti = pd.read_csv(cti_path)
cti["text"] = cti["text"].fillna("").astype(str)


def gold(entity_labels):
    if entity_labels is None or (isinstance(entity_labels, float) and np.isnan(entity_labels)):
        return 0
    labels = set(str(entity_labels).split("|")) - {""}
    return int(len(labels & IOC) > 0)


cti["y"] = cti["entity_labels"].apply(gold)
print(f"CTI rows {len(cti)}", flush=True)

ablation_rows = []
noise_rows = []
t0 = time.time()

for seed in SEEDS:
    print(f"\n===== IDS2018 seed {seed} =====", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["log_text"], df["y"], test_size=0.3, stratify=df["y"], random_state=seed
    )
    Xtr_list = X_train.tolist()
    Xte_list = X_test.tolist()
    ytr = y_train.to_numpy()
    yte = y_test.to_numpy()

    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
    Xtr_w = word.fit_transform(Xtr_list)
    Xte_w = word.transform(Xte_list)
    Xtr_c = char.fit_transform(Xtr_list)
    Xte_c = char.transform(Xte_list)
    zeros_tr = csr_matrix((Xtr_w.shape[0], 5))
    zeros_te = csr_matrix((Xte_w.shape[0], 5))
    Xtr_h = hstack([Xtr_c, zeros_tr])
    Xte_h = hstack([Xte_c, zeros_te])

    configs = {
        "Lexical-Word": (Xtr_w, Xte_w),
        "Structural-Char": (Xtr_c, Xte_c),
        "Semantic-Entity": (zeros_tr, zeros_te),
        "Hybrid-Robust": (Xtr_h, Xte_h),
    }
    trained = {}
    for config, (Xtr, Xte) in configs.items():
        trained[config] = train_keep(Xtr, ytr, seed)
        for name, model in trained[config].items():
            row = score(yte, model, Xte)
            row.update(
                {
                    "Seed": seed,
                    "Dataset": "IDS2018",
                    "Feature Config": config,
                    "Model": name,
                    "Noise Rate": 0.0,
                }
            )
            ablation_rows.append(row)
            print(
                f"  clean {config} {name} F1={row['F1_macro']:.4f}",
                flush=True,
            )

    for rate in NOISE_RATES:
        rng = random.Random(seed)
        noisy = Xte_list if rate == 0.0 else [apply_operational_noise(t, rate, rng) for t in Xte_list]
        mats = {
            "Lexical-Word": word.transform(noisy),
            "Structural-Char": char.transform(noisy),
            "Hybrid-Robust": hstack([char.transform(noisy), csr_matrix((len(noisy), 5))]),
        }
        for config, Xte in mats.items():
            for name, model in trained[config].items():
                row = score(yte, model, Xte)
                row.update(
                    {
                        "Seed": seed,
                        "Dataset": "IDS2018",
                        "Feature Config": config,
                        "Model": name,
                        "Noise Rate": rate,
                    }
                )
                noise_rows.append(row)
        print(f"  noise rate {rate:.2f} done", flush=True)

    print(f"\n===== CTI seed {seed} =====", flush=True)
    Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(
        cti["text"], cti["y"], test_size=0.3, stratify=cti["y"], random_state=seed
    )
    Xtr_t = Xtr_t.tolist()
    Xte_t = Xte_t.tolist()
    ytr_t = ytr_t.to_numpy()
    yte_t = yte_t.to_numpy()
    word_c = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
    char_c = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
    Xtr_wc = word_c.fit_transform(Xtr_t)
    Xte_wc = word_c.transform(Xte_t)
    Xtr_cc = char_c.fit_transform(Xtr_t)
    Xte_cc = char_c.transform(Xte_t)
    ent_tr = entity_matrix(Xtr_t)
    ent_te = entity_matrix(Xte_t)
    scaler = StandardScaler(with_mean=True)
    ent_tr_s = csr_matrix(scaler.fit_transform(ent_tr.astype(float)))
    ent_te_s = csr_matrix(scaler.transform(ent_te.astype(float)))
    for config, Xtr, Xte in (
        ("Lexical-Word", Xtr_wc, Xte_wc),
        ("Structural-Char", Xtr_cc, Xte_cc),
        ("Semantic-Entity", ent_tr_s, ent_te_s),
        ("Hybrid-Robust", hstack([Xtr_cc, ent_tr_s]), hstack([Xte_cc, ent_te_s])),
    ):
        fit_and_score(Xtr, Xte, ytr_t, yte_t, seed, "CTI", config, ablation_rows)

ablation = pd.DataFrame(ablation_rows)
noise = pd.DataFrame(noise_rows)
ablation.to_csv(os.path.join(ROOT, "results_multiseed_ablation.csv"), index=False)
noise.to_csv(os.path.join(ROOT, "results_multiseed_noise.csv"), index=False)


def summarize(frame, keys):
    ag = (
        frame.groupby(keys)[["Accuracy", "F1_macro", "AUC"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    ag.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in ag.columns
    ]
    return ag


summarize(ablation, ["Dataset", "Feature Config", "Model"]).to_csv(
    os.path.join(ROOT, "results_multiseed_ablation_summary.csv"), index=False
)
summarize(noise, ["Feature Config", "Model", "Noise Rate"]).to_csv(
    os.path.join(ROOT, "results_multiseed_noise_summary.csv"), index=False
)
print(f"Total {time.time() - t0:.1f}s", flush=True)
