"""Strings that carry both labels, and test rows that share port bin, protocol, and packet counts with training."""
import os
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT, "02-14-2018.csv")
df = pd.read_csv(path)
df = df.drop_duplicates().dropna()
df["Label"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)

dst = df["Dst Port"].lt(1024).map({True: "wellknown", False: "dynamic"})
raw = (
    "dst_" + dst
    + " proto_" + df["Protocol"].astype(str)
    + " dur_" + df["Flow Duration"].astype(str)
    + " fwd_" + df["Tot Fwd Pkts"].astype(str)
    + " bwd_" + df["Tot Bwd Pkts"].astype(str)
)

stemmer = PorterStemmer()
stop = set(stopwords.words("english"))


def preprocess_text(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return " ".join(stemmer.stem(w) for w in tokens if w not in stop)


df["log_text"] = raw.map(preprocess_text)
df["key"] = (
    dst.astype(str)
    + "|"
    + df["Protocol"].astype(str)
    + "|"
    + df["Tot Fwd Pkts"].astype(str)
    + "|"
    + df["Tot Bwd Pkts"].astype(str)
)
df["dur"] = df["Flow Duration"].astype("int64")

# Conflicting labels on the serialized string, before (text, label) dedup.
g = df.groupby("log_text")["Label"].nunique()
both = g[g > 1]
print("strings_with_both_labels", int(both.shape[0]))
print("rows_in_those_strings", int(df["log_text"].isin(both.index).sum()))

dedup = df.drop_duplicates(subset=["log_text", "Label"]).copy()
print("rows_after_text_label_dedup", len(dedup))
print(dedup["Label"].value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    dedup["log_text"],
    dedup["Label"],
    test_size=0.3,
    stratify=dedup["Label"],
    random_state=42,
)
train = dedup.loc[X_train.index]
test = dedup.loc[X_test.index]

# Same exact string on both sides of the split (only possible if labels differ).
overlap = set(train["log_text"]) & set(test["log_text"])
print("exact_strings_in_train_and_test", len(overlap))

# Near duplicates: same port bin, protocol, and packet counts; duration may differ.
train_keys = set(train["key"])
test_hit = test["key"].isin(train_keys)
print("test_rows", len(test))
print("test_rows_key_seen_in_train", int(test_hit.sum()))
print("test_rows_key_frac", float(test_hit.mean()))

# Tighter: for matching keys, closest train duration within 1% or 1e5 microseconds.
train_by_key = train.groupby("key")["dur"].apply(lambda s: s.to_numpy())
close_1pct = 0
close_1e5 = 0
exact_dur = 0
checked = 0
for key, dur in zip(test.loc[test_hit, "key"], test.loc[test_hit, "dur"]):
    arr = train_by_key[key]
    checked += 1
    diff = abs(arr - dur)
    exact_dur += int((diff == 0).any())
    close_1e5 += int((diff <= 100_000).any())
    # relative 1%, with a floor so zero-duration matches stay exact-only
    rel = diff <= (0.01 * max(abs(int(dur)), 1))
    close_1pct += int(rel.any())

print("checked_test_rows_with_shared_key", checked)
print("shared_key_and_exact_duration", exact_dur)
print("shared_key_duration_within_1e5_us", close_1e5)
print("shared_key_duration_within_1pct", close_1pct)
