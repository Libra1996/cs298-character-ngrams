"""How many characters the noise routine changes on the seed-42 test strings.

An edit is one operation that changes the string: a typo substitution,
deletion, or insertion; a delimiter replacement whose new text differs from
the original character; or an inserted fragment space. The noise generator
is reset to seed 42 at each rate, matching the published noise run.
"""
import os
import random
import re
import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
ALPHANUM = string.ascii_lowercase + string.digits
RATES = (0.05, 0.10, 0.20)


def count_edits(text, rate, rng):
    typos = delims = frags = 0
    out = []
    for ch in text:
        if rng.random() < rate:
            op = rng.choice(("sub", "del", "ins"))
            if op == "sub":
                repl = rng.choice(ALPHANUM)
                out.append(repl)
                typos += int(repl != ch)
            elif op == "del":
                typos += 1
            else:
                out.append(ch)
                out.append(rng.choice(ALPHANUM))
                typos += 1
        else:
            out.append(ch)
    s = "".join(out)
    delim_out = []
    for ch in s:
        if ch in (" ", "_") and rng.random() < rate:
            repl = rng.choice((" ", "  ", "_", "-"))
            delim_out.append(repl)
            delims += int(repl != ch)
        else:
            delim_out.append(ch)
    s = "".join(delim_out)
    frag = []
    i = 0
    while i < len(s):
        ch = s[i]
        frag.append(ch)
        if ch.isalnum() and i + 1 < len(s) and s[i + 1].isalnum() and rng.random() < rate:
            frag.append(" ")
            frags += 1
        i += 1
    return typos, delims, frags


print("Loading...", flush=True)
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
_, X_test = train_test_split(
    df["log_text"], test_size=0.3, stratify=df["y"], random_state=42
)
texts = X_test.tolist()
lengths = np.array([len(t) for t in texts])
print(f"test {len(texts)} mean_len {lengths.mean():.2f}", flush=True)

rows = []
for rate in RATES:
    rng = random.Random(42)
    counts = np.array([count_edits(t, rate, rng) for t in texts], dtype=float)
    total = counts.sum(axis=1)
    rows.append(
        {
            "rate": rate,
            "n": len(texts),
            "mean_len": float(lengths.mean()),
            "mean_typos": float(counts[:, 0].mean()),
            "mean_delims": float(counts[:, 1].mean()),
            "mean_frags": float(counts[:, 2].mean()),
            "mean_edits": float(total.mean()),
            "std_edits": float(total.std(ddof=1)),
            "mean_edits_per_char": float((total / lengths).mean()),
        }
    )
    print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results_noise_edit_counts.csv"), index=False)
print("wrote results_noise_edit_counts.csv", flush=True)
