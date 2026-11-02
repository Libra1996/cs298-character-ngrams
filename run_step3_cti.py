"""Step 3 (CTI): Semantic-Entity regex features on annotated CTI text."""
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
CTI_CSV = os.path.join(ROOT, "data", "cti", "cyber_threat_intelligence.csv")
if not os.path.isfile(CTI_CSV):
    raise FileNotFoundError(CTI_CSV)

print(f"Loading: {CTI_CSV}")
t0 = time.time()
df = pd.read_csv(CTI_CSV)
df["text"] = df["text"].fillna("").astype(str)

# Gold IoC-ish labels from Schwankner annotations (for supervised binary task)
IOC_GOLD = {
    "IPV4",
    "URL",
    "DOMAIN",
    "MD5",
    "SHA1",
    "SHA2",
    "FILEPATH",
    "REGISTRYKEY",
    "EMAIL",
    "malware",
    "vulnerability",
    "threat-actor",
    "attack-pattern",
    "campaign",
    "tools",
}


def gold_ioc_label(entity_labels) -> int:
    if entity_labels is None or (isinstance(entity_labels, float) and np.isnan(entity_labels)):
        return 0
    labels = set(str(entity_labels).split("|")) - {""}
    return int(len(labels & IOC_GOLD) > 0)


df["Label"] = df["entity_labels"].apply(gold_ioc_label)
print("Label distribution (1=IoC-bearing CTI snippet):")
print(df["Label"].value_counts())

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
    "count_hash": re.compile(
        r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b"
    ),
    "count_path": re.compile(
        r"(?:[A-Za-z]:\\(?:[^\s\\/:*?\"<>|\r\n]+\\)*[^\s\\/:*?\"<>|\r\n]+)"
        r"|(?:/(?:(?:etc|var|usr|home|tmp|opt|bin|sbin)/|\S+/)+)\S+"
    ),
}


def extract_entity_counts(text: str) -> dict:
    return {name: len(pat.findall(text)) for name, pat in ENTITY_PATTERNS.items()}


print("Extracting regex entity counts...")
t1 = time.time()
entity_df = pd.DataFrame(list(df["text"].apply(extract_entity_counts)))
feature_cols = list(ENTITY_PATTERNS.keys())
print(f"Extraction time: {time.time() - t1:.1f}s")
print("\nRegex entity count means (CTI text):")
print(entity_df[feature_cols].mean())
print("Entity count maxima:")
print(entity_df[feature_cols].max())
print(
    "Rows with any regex entity > 0:",
    int((entity_df[feature_cols].sum(axis=1) > 0).sum()),
    "/",
    len(df),
)

X_train, X_test, y_train, y_test = train_test_split(
    entity_df[feature_cols],
    df["Label"],
    test_size=0.3,
    stratify=df["Label"],
    random_state=42,
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.astype(float))
X_test_s = scaler.transform(X_test.astype(float))
print(f"Train feature variance sum: {np.var(X_train_s, axis=0).sum():.6f}")

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
    print(f"\nTraining {name}...")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC: {auc:.4f}")
    rows.append(
        {
            "Dataset": "CTI (mrmoor)",
            "Feature Config": "Semantic-Entity (regex counts)",
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall (macro)": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "AUC-ROC": auc,
        }
    )

results = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "results_step3_semantic_entity_cti.csv")
results.to_csv(out_csv, index=False)
print("\n=== Semantic-Entity on CTI — Summary ===")
print(results.round(4).to_string(index=False))
print(f"Saved: {out_csv}")

diag = {
    "dataset": "mrmoor/cyber-threat-intelligence",
    "rows_total": len(df),
    "label_1_ioc_bearing": int((df["Label"] == 1).sum()),
    "label_0_other": int((df["Label"] == 0).sum()),
    "rows_with_any_regex_entity": int((entity_df[feature_cols].sum(axis=1) > 0).sum()),
    **{f"mean_{c}": float(entity_df[c].mean()) for c in feature_cols},
    **{f"max_{c}": int(entity_df[c].max()) for c in feature_cols},
}
diag_path = os.path.join(ROOT, "results_step3_semantic_entity_cti_diagnostics.csv")
pd.DataFrame([diag]).to_csv(diag_path, index=False)
print(f"Saved: {diag_path}")
print(f"Total time: {time.time() - t0:.1f}s")
