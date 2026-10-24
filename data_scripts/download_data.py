"""
Phase 2 - Data Collection
Downloads the three public datasets used by SentiBias Audit into data/raw/
and writes a download log (URL, status, size, timestamp) as evidence.

Datasets:
  1. Roman Urdu Sentiment (UCI ML Repository, dataset #458)
  2. TweetEval Sentiment (English tweets, Cardiff NLP, GitHub)
  3. Equity Evaluation Corpus (EEC, gender/race-coded names, Kiritchenko & Mohammad 2018)
"""
import os
import zipfile
from datetime import datetime

import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
LOG_FILE = os.path.join(RAW_DIR, "download_log.txt")

SOURCES = {
    # Roman Urdu: UCI #458 (text, label[Positive/Negative/Neutral])
    "roman_urdu.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00458/Roman%20Urdu%20DataSet.csv",
    # TweetEval sentiment: 0=negative, 1=neutral, 2=positive
    "tweeteval_train_text.txt": "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/train_text.txt",
    "tweeteval_train_labels.txt": "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/train_labels.txt",
    "tweeteval_val_text.txt": "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/val_text.txt",
    "tweeteval_val_labels.txt": "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/val_labels.txt",
    "tweeteval_test_text.txt": "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_text.txt",
    "tweeteval_test_labels.txt": "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_labels.txt",
    # EEC: 8,640 template sentences with gender/race-coded names
    "eec.zip": "https://saifmohammad.com/WebDocs/EEC/Equity-Evaluation-Corpus.zip",
}


def log(message):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def download(filename, url):
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path):
        log(f"SKIP   {filename} already exists ({os.path.getsize(path)} bytes)")
        return path
    # Some hosts (e.g. saifmohammad.com) return HTTP 406 for the default
    # python-requests user agent, so send a browser-style one.
    headers = {"User-Agent": "Mozilla/5.0 (SentiBias-Audit data collection)"}
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    log(f"OK     {filename} <- {url} (HTTP {response.status_code}, {len(response.content)} bytes)")
    return path


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    log("=== download_data.py started ===")
    for filename, url in SOURCES.items():
        download(filename, url)

    # Unzip the EEC archive (contains Equity-Evaluation-Corpus.csv)
    eec_zip = os.path.join(RAW_DIR, "eec.zip")
    with zipfile.ZipFile(eec_zip) as z:
        z.extractall(RAW_DIR)
        log(f"UNZIP  eec.zip -> {z.namelist()}")
    log("=== download_data.py finished ===")


if __name__ == "__main__":
    main()
