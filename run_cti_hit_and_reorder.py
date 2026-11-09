"""CTI regex-hit test rows, and token shuffle on seeds 42, 7, 13, 21, and 99."""
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 7, 13, 21, 99]
IOC_GOLD = {
    "IPV4", "URL", "DOMAIN", "MD5", "SHA1", "SHA2", "FILEPATH",
    "REGISTRYKEY", "EMAIL", "malware", "vulnerability", "threat-actor",
    "attack-pattern", "campaign", "tools",
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


def tags_of(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return set(str(value).split("|")) - {""}


def counts(text):
    return [len(pat.findall(text)) for pat in ENTITY_PATTERNS.values()]


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
    return [("LR", lr), ("RF", rf), ("Ensemble", ens)]


def cti_hit_rows():
    df = pd.read_csv(os.path.join(ROOT, "data", "cti", "cyber_threat_intelligence.csv"))
    df["text"] = df["text"].fillna("").astype(str)
    texts = df["text"].to_numpy()
    y = df["entity_labels"].map(lambda v: int(len(tags_of(v) & IOC_GOLD) > 0)).to_numpy()
    entity = np.vstack([counts(t) for t in texts]).astype(float)
    hit = entity.sum(axis=1) > 0
    rows = []
    for seed in SEEDS:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
        te_hit = te[hit[te]]
        word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
        char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
        Xw_tr, Xw_te = word.fit_transform(texts[tr]), word.transform(texts[te_hit])
        Xc_tr, Xc_te = char.fit_transform(texts[tr]), char.transform(texts[te_hit])
        scaler = StandardScaler(with_mean=True)
        Xe_tr = scaler.fit_transform(entity[tr])
        Xe_te = scaler.transform(entity[te_hit])
        blocks = {
            "Lexical-Word": (Xw_tr, Xw_te),
            "Structural-Char": (Xc_tr, Xc_te),
            "Semantic-Entity": (Xe_tr, Xe_te),
            "Hybrid-Robust": (hstack([Xc_tr, csr_matrix(Xe_tr)]), hstack([Xc_te, csr_matrix(Xe_te)])),
        }
        ytr, yte = y[tr], y[te_hit]
        print(
            f"seed {seed} regex-hit test {len(te_hit)} pos {int(yte.sum())}",
            flush=True,
        )
        for config, (A, B) in blocks.items():
            for name, model in make_models(seed):
                model.fit(A, ytr)
                pred = model.predict(B)
                proba = model.predict_proba(B)[:, 1]
                row = {
                    "seed": seed,
                    "config": config,
                    "model": name,
                    "n": int(len(yte)),
                    "pos": int(yte.sum()),
                    "acc": float(accuracy_score(yte, pred)),
                    "f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                    "auc": float(roc_auc_score(yte, proba)) if len(np.unique(yte)) > 1 else float("nan"),
                }
                rows.append(row)
                print(
                    f"  {config} {name} f1 {row['f1']:.4f} acc {row['acc']:.4f} auc {row['auc']:.4f}",
                    flush=True,
                )
    out = os.path.join(ROOT, "results_cti_regex_hit_test.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, flush=True)


def reorder_tokens(texts, seed):
    rng = np.random.RandomState(seed)
    out = []
    for text in texts:
        toks = str(text).split()
        rng.shuffle(toks)
        out.append(" ".join(toks))
    return out


def reorder_rows():
    df = pd.read_pickle(os.path.join(ROOT, "cache_dedup_grouped.pkl"))
    rows = []
    for seed in SEEDS:
        tr, te = train_test_split(
            df.index, test_size=0.3, stratify=df["y"], random_state=seed
        )
        train = df.loc[tr]
        test = df.loc[te]
        ytr = train["y"].to_numpy()
        yte = test["y"].to_numpy()
        shuffled = reorder_tokens(test["log_text"].tolist(), seed)
        for feature, analyzer, ngram in (
            ("Lexical-Word", "word", (1, 2)),
            ("Structural-Char", "char", (3, 5)),
        ):
            vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=10000)
            Xtr = vec.fit_transform(train["log_text"])
            clean = vec.transform(test["log_text"])
            noisy = vec.transform(shuffled)
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            lr.fit(Xtr, ytr)
            for kind, Xte in (("clean", clean), ("reorder", noisy)):
                pred = lr.predict(Xte)
                f1 = float(f1_score(yte, pred, average="macro"))
                rows.append({"seed": seed, "feature": feature, "kind": kind, "f1_macro": f1})
                print(f"seed {seed} {feature} {kind} {f1:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results_token_reorder_seeds.csv"), index=False)
    print("wrote results_token_reorder_seeds.csv", flush=True)


if __name__ == "__main__":
    cti_hit_rows()
    reorder_rows()
