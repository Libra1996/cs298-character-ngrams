"""Snippet checks: tags the five patterns miss, repeated test rows, and the regex-hit subset."""
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
IOC_GOLD = {
    "IPV4", "URL", "DOMAIN", "MD5", "SHA1", "SHA2", "FILEPATH",
    "REGISTRYKEY", "EMAIL", "malware", "vulnerability", "threat-actor",
    "attack-pattern", "campaign", "tools",
}
REGEX_TAGS = {"IPV4", "URL", "DOMAIN", "MD5", "SHA1", "SHA2", "FILEPATH"}
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
COLS = list(ENTITY_PATTERNS.keys())


def tags_of(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return set(str(value).split("|")) - {""}


def gold(value):
    return int(len(tags_of(value) & IOC_GOLD) > 0)


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


def pack(texts, labels, entity):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
    Xw = word.fit_transform(texts[0]), word.transform(texts[1])
    Xc = char.fit_transform(texts[0]), char.transform(texts[1])
    scaler = StandardScaler(with_mean=True)
    Xe = scaler.fit_transform(entity[0]), scaler.transform(entity[1])
    Xh = (
        hstack([Xc[0], csr_matrix(Xe[0])]),
        hstack([Xc[1], csr_matrix(Xe[1])]),
    )
    return {"Lexical-Word": Xw, "Structural-Char": Xc, "Semantic-Entity": Xe, "Hybrid-Robust": Xh}


def score_one(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")
    return {
        "n": int(len(yte)),
        "pos": int(np.sum(yte)),
        "acc": accuracy_score(yte, pred),
        "f1": f1_score(yte, pred, average="macro", zero_division=0),
        "auc": auc,
    }


def run_split(texts, y, entity, seed, mask_test=None, tag="full"):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    if mask_test is not None:
        te = te[mask_test[te]]
    mats = pack(
        (texts[tr], texts[te]),
        y,
        (entity[tr], entity[te]),
    )
    rows = []
    for config, (Xtr, Xte) in mats.items():
        for name, model in make_models(seed):
            stats = score_one(model, Xtr, Xte, y[tr], y[te])
            stats.update({"tag": tag, "config": config, "model": name, "seed": seed})
            rows.append(stats)
            print(
                f"{tag} {config} {name} n={stats['n']} pos={stats['pos']} "
                f"f1={stats['f1']:.4f} acc={stats['acc']:.4f} auc={stats['auc']:.4f}",
                flush=True,
            )
    return rows, tr, te


df = pd.read_csv(os.path.join(ROOT, "data", "cti", "cyber_threat_intelligence.csv"))
df["text"] = df["text"].fillna("").astype(str)
raw = df["entity_labels"]
missing = int(raw.isna().sum())
tag_sets = raw.map(tags_of)
positive = tag_sets.map(lambda s: len(s & IOC_GOLD) > 0)
only_nonregex = tag_sets.map(lambda s: len(s & IOC_GOLD) > 0 and len(s & REGEX_TAGS) == 0)
print(f"rows {len(df)} missing {missing} positive {int(positive.sum())} only_nonregex {int(only_nonregex.sum())}")

texts = df["text"].to_numpy()
y = raw.map(gold).to_numpy()
entity = np.vstack(df["text"].map(counts).to_list()).astype(float)
hit = entity.sum(axis=1) > 0
print(f"regex hit {int(hit.sum())} hit positive {int(y[hit].sum())} hit negative {int((y[hit] == 0).sum())}")

rows = []
full_rows, tr, te = run_split(texts, y, entity, 42, tag="full")
rows.extend(full_rows)
train_text = set(texts[tr])
novel = np.array([t not in train_text for t in texts])
print(f"test {len(te)} repeated {int((~novel[te]).sum())} kept {int(novel[te].sum())}")
novel_rows, _, _ = run_split(texts, y, entity, 42, mask_test=novel, tag="no_repeat")
rows.extend(novel_rows)

sub_text, sub_y, sub_ent = texts[hit], y[hit], entity[hit]
print(f"subset rows {len(sub_y)} pos {int(sub_y.sum())}")
sub_rows, _, _ = run_split(sub_text, sub_y, sub_ent, 42, tag="regex_subset")
rows.extend(sub_rows)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results_issue6_cti.csv"), index=False)
print("done", flush=True)
