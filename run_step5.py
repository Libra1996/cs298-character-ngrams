"""Four representations times logistic regression, the forest, and the soft vote.

The flow rows are read back from the step 1–4 result files. Word and
character models for the snippet corpus are trained in this script.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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

ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIG_ORDER = [
    "Lexical-Word (TF-IDF 1-2)",
    "Structural-Char (TF-IDF 3-5)",
    "Semantic-Entity (regex counts)",
    "Hybrid-Robust (char TF-IDF + entity)",
]
MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "Soft-Voting Ensemble",
]


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


def eval_models(X_train, X_test, y_train, y_test, dataset, config_name):
    rows = []
    for name, model in make_models().items():
        print(f"  Training {name} ({config_name})...")
        t = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"  Train+predict: {time.time() - t:.1f}s")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        auc = roc_auc_score(y_test, y_proba)
        print(f"  AUC-ROC: {auc:.4f}\n")
        rows.append(
            {
                "Dataset": dataset,
                "Feature Config": config_name,
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
    return pd.DataFrame(rows)


def load_tagged(path, dataset, config_name=None):
    df = pd.read_csv(path)
    if "Dataset" not in df.columns or df["Dataset"].isna().all():
        df["Dataset"] = dataset
    else:
        df["Dataset"] = df["Dataset"].fillna(dataset)
    if config_name is not None:
        df["Feature Config"] = config_name
    cols = [
        "Dataset",
        "Feature Config",
        "Model",
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1 (macro)",
        "AUC-ROC",
    ]
    return df[cols]


def gold_ioc_label(entity_labels) -> int:
    ioc = {
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
    if entity_labels is None or (
        isinstance(entity_labels, float) and np.isnan(entity_labels)
    ):
        return 0
    labels = set(str(entity_labels).split("|")) - {""}
    return int(len(labels & ioc) > 0)


def sort_ablation(df):
    ds_order = {"IDS2018": 0, "CTI (mrmoor)": 1}
    df = df.copy()
    df["_ds"] = df["Dataset"].map(ds_order).fillna(99)
    df["_cfg"] = df["Feature Config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    df["_mdl"] = df["Model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values(["_ds", "_cfg", "_mdl"]).drop(columns=["_ds", "_cfg", "_mdl"])
    return df.reset_index(drop=True)


t0 = time.time()

# ----- Assemble IDS2018 12-run table from Steps 1-4 -----
ids_frames = [
    load_tagged(
        os.path.join(ROOT, "results_step1_lexical_word_baseline.csv"), "IDS2018"
    ),
    load_tagged(os.path.join(ROOT, "results_step2_structural_char.csv"), "IDS2018"),
    load_tagged(
        os.path.join(ROOT, "results_step3_semantic_entity_ids2018.csv"), "IDS2018"
    ),
]
hybrid = pd.read_csv(os.path.join(ROOT, "results_step4_hybrid_robust.csv"))
ids_frames.append(
    hybrid.loc[hybrid["Dataset"] == "IDS2018"][
        [
            "Dataset",
            "Feature Config",
            "Model",
            "Accuracy",
            "Precision (macro)",
            "Recall (macro)",
            "F1 (macro)",
            "AUC-ROC",
        ]
    ]
)
ids2018 = pd.concat(ids_frames, ignore_index=True)
print(f"IDS2018 ablation rows: {len(ids2018)} (expect 12)")

# ----- CTI: reuse entity + hybrid; train missing word and char -----
cti_csv = os.path.join(ROOT, "data", "cti", "cyber_threat_intelligence.csv")
cti = pd.read_csv(cti_csv)
cti["text"] = cti["text"].fillna("").astype(str)
cti["Label"] = cti["entity_labels"].apply(gold_ioc_label)

X_train, X_test, y_train, y_test = train_test_split(
    cti["text"],
    cti["Label"],
    test_size=0.3,
    stratify=cti["Label"],
    random_state=42,
)

print("\n========== CTI Lexical-Word (TF-IDF 1-2) ==========")
tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
X_train_w = tfidf_word.fit_transform(X_train)
X_test_w = tfidf_word.transform(X_test)
print(f"Word vocab: {len(tfidf_word.vocabulary_)}")
cti_word = eval_models(
    X_train_w, X_test_w, y_train, y_test, "CTI (mrmoor)", "Lexical-Word (TF-IDF 1-2)"
)

print("========== CTI Structural-Char (TF-IDF 3-5) ==========")
tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
X_train_c = tfidf_char.fit_transform(X_train)
X_test_c = tfidf_char.transform(X_test)
print(f"Char vocab: {len(tfidf_char.vocabulary_)}")
cti_char = eval_models(
    X_train_c,
    X_test_c,
    y_train,
    y_test,
    "CTI (mrmoor)",
    "Structural-Char (TF-IDF 3-5)",
)

cti_entity = load_tagged(
    os.path.join(ROOT, "results_step3_semantic_entity_cti.csv"), "CTI (mrmoor)"
)
cti_hybrid = hybrid.loc[hybrid["Dataset"] == "CTI (mrmoor)"][
    [
        "Dataset",
        "Feature Config",
        "Model",
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1 (macro)",
        "AUC-ROC",
    ]
]
cti_all = pd.concat([cti_word, cti_char, cti_entity, cti_hybrid], ignore_index=True)
print(f"CTI ablation rows: {len(cti_all)} (expect 12)")

# ----- Combined export -----
ablation = sort_ablation(pd.concat([ids2018, cti_all], ignore_index=True))
out_csv = os.path.join(ROOT, "results_step5_ablation_table.csv")
ablation.to_csv(out_csv, index=False)

rounded = ablation.copy()
for col in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "AUC-ROC"]:
    rounded[col] = rounded[col].round(4)
rounded_csv = os.path.join(ROOT, "results_step5_ablation_table_rounded.csv")
rounded.to_csv(rounded_csv, index=False)

print("\n=== Step 5 — Full ablation (24 rows: 2 datasets x 12 runs) ===")
print(rounded.to_string(index=False))
print(f"Saved: {out_csv}")
print(f"Saved: {rounded_csv}")
print(f"Total time: {time.time() - t0:.1f}s")
