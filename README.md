# Character n-grams and entity counts on one day of CSE-CIC-IDS2018

These are the scripts for my CS 298 project. Logistic regression, a random forest of depth 10, and their unweighted soft vote stay fixed. What changes is how a flow is written as text: word TF-IDF, character n-grams of length 3 to 5, five regex counts (IP, URL or domain, CVE, hash, file path), or the character vector with those counts appended.

The flow file is February 14, 2018, from CSE-CIC-IDS2018. The snippet files under `data/cti/` are a separate check. They are never mixed into the flow training set.

## Data file not include in git

`02-14-2018.csv` is about 358 MB, which is over GitHub's limit for a normal file. Download the February 14 day file from the University of New Brunswick and put it in this folder, next to the scripts:

https://www.unb.ca/cic/datasets/ids-2018.html

The snippet export is already here. It is Alexander Schwankner's public corpus (`mrmoor/cyber-threat-intelligence` on Hugging Face), CC BY 4.0. See `data/cti/CITATION.txt`.

## Setup

I ran these with Python 3.12.10. The versions I had installed:

```
pip install -r requirements.txt
```

The first script that stems text will download the NLTK stopword list.

## What to run

`run_step1.py` through `run_step7.py` are the main pass.

| Script | What it writes |
|---|---|
| `run_step1.py` | Word TF-IDF |
| `run_step2.py` | Character TF-IDF |
| `run_step3.py` | Five regex counts on the flow strings |
| `run_step3_cti.py` | The same counts on the snippets |
| `run_step4.py` | Character vector plus the five counts |
| `run_step5.py` | The 12-run ablation table |
| `run_step6.py` | Character noise on the test strings only |
| `run_step7.py` | Latency, serialized size, and traced memory |

The other scripts are the checks that came after that pass: the numeric columns, duration bins, five seeds, the 10:00 cut, per-class scores, one-core timing, the resident set, how many characters the noise routine changes, shared port and packet-count keys, the hour split, the grouped split, the depth sweep, and the snippet checks (`run_issue2_*.py` through `run_issue6_cti.py`).

The `results_*.csv` files are the runs I used in the thesis. Latency will not match those files on another machine. The detection scores should, if the day file and the library versions match.

The split is stratified 70/30 with seed 42, after duplicate rows, missing values, and duplicate text-and-label pairs are dropped.
