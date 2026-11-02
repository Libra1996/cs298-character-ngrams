"""Character noise on the February 14 test strings.

Train on the clean strings. Corrupt only the test fold: typos, delimiter
changes, and spaces inserted inside tokens. Word, character, and hybrid
are scored. The entity counts on these flows are all zero, so that
configuration is skipped.
"""
import os
import random
import re
import string
import time

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords", quiet=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
NOISE_RATES = (0.0, 0.05, 0.10, 0.20)
ALPHANUM = string.ascii_lowercase + string.digits

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
FEATURE_COLS = list(ENTITY_PATTERNS.keys())


def extract_entity_counts(text: str) -> dict:
    return {name: len(pat.findall(text)) for name, pat in ENTITY_PATTERNS.items()}


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


def apply_operational_noise(text: str, rate: float, rng: random.Random) -> str:
    """Mix typos, delimiter drift, and token fragmentation at one severity rate."""
    if rate <= 0 or not text:
        return text
    chars = list(text)

    # Typos: substitute, delete, or insert a character.
    out = []
    for ch in chars:
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

    # 2) Irregular delimiters: spaces / underscores swap or extra whitespace
    delim_out = []
    for ch in s:
        if ch in (" ", "_") and rng.random() < rate:
            delim_out.append(rng.choice((" ", "  ", "_", "-")))
        else:
            delim_out.append(ch)
    s = "".join(delim_out)

    # 3) Fragmented tokens: insert a space inside some alphanumeric runs
    frag = []
    i = 0
    while i < len(s):
        ch = s[i]
        frag.append(ch)
        if ch.isalnum() and i + 1 < len(s) and s[i + 1].isalnum() and rng.random() < rate:
            frag.append(" ")
        i += 1
    return "".join(frag)


def serialize_log_balanced(row):
    dst_type = "wellknown" if row["Dst Port"] < 1024 else "dynamic"
    return (
        f"dst_{dst_type} proto_{row['Protocol']} "
        f"dur_{row['Flow Duration']} fwd_{row['Tot Fwd Pkts']} bwd_{row['Tot Bwd Pkts']}"
    )


def preprocess_text(text, stop_words, stemmer):
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


def score_row(y_true, y_pred, y_proba, dataset, config, model, rate):
    return {
        "Dataset": dataset,
        "Feature Config": config,
        "Model": model,
        "Noise Rate": rate,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
    }


t0 = time.time()
file_path = next(
    (
        p
        for p in [
            os.path.join(ROOT, "02-14-2018.csv"),
        ]
        if os.path.isfile(p)
    ),
    None,
)
if file_path is None:
    raise FileNotFoundError("02-14-2018.csv not found")

print(f"Loading: {file_path}")
df = pd.read_csv(file_path)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["Label"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)

print("Serializing + preprocessing (same as Steps 1-4)...")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
df["log_text"] = df.apply(serialize_log_balanced, axis=1)
df["log_text"] = df["log_text"].apply(lambda t: preprocess_text(t, stop_words, stemmer))
df.drop_duplicates(subset=["log_text", "Label"], inplace=True)
print(f"Rows after dedup: {len(df)}")

X_train, X_test, y_train, y_test = train_test_split(
    df["log_text"],
    df["Label"],
    test_size=0.3,
    stratify=df["Label"],
    random_state=42,
)
X_train_list = X_train.tolist()
X_test_clean = X_test.tolist()
y_test = y_test.to_numpy()
y_train = y_train.to_numpy()

# Fit vectorizers on clean training text only
print("Fitting word / char TF-IDF on clean training logs...")
tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
X_train_word = tfidf_word.fit_transform(X_train_list)
X_train_char = tfidf_char.fit_transform(X_train_list)

ent_train = pd.DataFrame([extract_entity_counts(t) for t in X_train_list])[FEATURE_COLS]
scaler = StandardScaler(with_mean=True)
ent_train_s = scaler.fit_transform(ent_train.astype(float))
X_train_hybrid = hstack([X_train_char, csr_matrix(ent_train_s)])

print("Training models on clean data...")
trained = {}
for config, Xtr in [
    ("Lexical-Word (TF-IDF 1-2)", X_train_word),
    ("Structural-Char (TF-IDF 3-5)", X_train_char),
    ("Hybrid-Robust (char TF-IDF + entity)", X_train_hybrid),
]:
    trained[config] = {}
    for name, model in make_models().items():
        t = time.time()
        model.fit(Xtr, y_train)
        trained[config][name] = model
        print(f"  {config} / {name}: {time.time() - t:.1f}s")

# Example strings for the report
rng_ex = random.Random(0)
examples = []
sample = X_test_clean[0]
for rate in NOISE_RATES:
    examples.append(
        {
            "Noise Rate": rate,
            "Original": sample,
            "Noisy": apply_operational_noise(sample, rate, rng_ex),
        }
    )
ex_path = os.path.join(ROOT, "results_step6_noise_examples.csv")
pd.DataFrame(examples).to_csv(ex_path, index=False)
print(f"Saved: {ex_path}")

rows = []
for rate in NOISE_RATES:
    print(f"\n===== Evaluate at noise rate {rate:.2f} =====")
    rng = random.Random(42)
    if rate == 0.0:
        noisy = X_test_clean
    else:
        noisy = [apply_operational_noise(t, rate, rng) for t in X_test_clean]

    X_test_word = tfidf_word.transform(noisy)
    X_test_char = tfidf_char.transform(noisy)
    ent_test = pd.DataFrame([extract_entity_counts(t) for t in noisy])[FEATURE_COLS]
    ent_test_s = scaler.transform(ent_test.astype(float))
    X_test_hybrid = hstack([X_test_char, csr_matrix(ent_test_s)])
    print(
        f"  Regex entities on noisy test >0: "
        f"{int((ent_test.sum(axis=1) > 0).sum())} / {len(noisy)}"
    )

    feature_mats = {
        "Lexical-Word (TF-IDF 1-2)": X_test_word,
        "Structural-Char (TF-IDF 3-5)": X_test_char,
        "Hybrid-Robust (char TF-IDF + entity)": X_test_hybrid,
    }
    for config, Xte in feature_mats.items():
        for name, model in trained[config].items():
            y_pred = model.predict(Xte)
            y_proba = model.predict_proba(Xte)[:, 1]
            row = score_row(y_test, y_pred, y_proba, "IDS2018", config, name, rate)
            rows.append(row)
            print(
                f"  {config} | {name}: "
                f"F1={row['F1 (macro)']:.4f} AUC={row['AUC-ROC']:.4f}"
            )

results = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "results_step6_noise_robustness.csv")
results.to_csv(out_csv, index=False)

rounded = results.copy()
for col in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "AUC-ROC"]:
    rounded[col] = rounded[col].round(4)
rounded_csv = os.path.join(ROOT, "results_step6_noise_robustness_rounded.csv")
rounded.to_csv(rounded_csv, index=False)

print("\n=== Step 6 — Noise robustness (IDS2018) ===")
print(rounded.to_string(index=False))
print(f"Saved: {out_csv}")
print(f"Saved: {rounded_csv}")
print(f"Total time: {time.time() - t0:.1f}s")
