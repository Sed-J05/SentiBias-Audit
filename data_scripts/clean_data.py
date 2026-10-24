"""
Phase 2 - Cleaning Implementation
Reads the raw files from data/raw/, handles missing values, removes duplicates,
manages outliers, and writes one tidy CSV per dataset plus a combined master
file to data/processed/. A JSON summary of every cleaning step is also saved.

Unified schema of the processed files:
  text, label (negative/neutral/positive), language, source, split,
  gender, code_switched, n_tokens, n_chars, english_word_ratio
"""
import json
import os
import re

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "processed")

# Small lexicon of frequent, unambiguous English words used to detect
# code-switching inside Roman Urdu text (ambiguous tokens such as "a", "to",
# "me", "he", "is", "so" are deliberately excluded because they are also
# valid Roman Urdu words).
ENGLISH_WORDS = {
    "the", "this", "that", "with", "your", "you", "are", "have", "was", "were",
    "not", "but", "and", "for", "from", "about", "just", "now", "still", "all",
    "good", "bad", "nice", "love", "great", "best", "worst", "happy", "sad",
    "please", "thanks", "thank", "awesome", "amazing", "beautiful", "wrong",
    "right", "true", "fake", "work", "life", "day", "night", "friend", "people",
    "problem", "support", "only", "again", "always", "never", "think", "know",
    "want", "need", "feel", "miss", "wait", "hope", "what", "why", "when",
    "match", "team", "player", "cricket", "game", "win", "lost", "like",
    "hate", "well", "done", "really", "very", "much", "super", "movie", "song",
    "video", "time", "new", "old", "big", "more", "most", "some", "many",
    "every", "last", "first", "next", "back", "out", "over", "before", "after",
}

SUMMARY = {}


def tokenize(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def english_ratio(text):
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(t in ENGLISH_WORDS for t in tokens) / len(tokens)


def basic_clean(df, name):
    """Shared steps: missing values, whitespace, duplicates, outliers."""
    steps = {"raw_rows": int(len(df))}

    # 1. Missing values: drop rows with no text or no label
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    df = df[df["text"] != ""]
    steps["after_missing"] = int(len(df))

    # 2. Duplicates: keep the first occurrence of each text
    df = df.drop_duplicates(subset=["text"])
    steps["after_duplicates"] = int(len(df))

    # 3. Outliers: token-count based, Tukey outer fence (Q3 + 3*IQR) and
    #    minimum of 1 token
    df["n_tokens"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()
    q1, q3 = df["n_tokens"].quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)
    df = df[(df["n_tokens"] >= 1) & (df["n_tokens"] <= upper)]
    steps["outlier_token_fence"] = float(upper)
    steps["after_outliers"] = int(len(df))

    steps["label_counts"] = df["label"].value_counts().to_dict()
    SUMMARY[name] = steps
    return df


def clean_roman_urdu():
    df = pd.read_csv(os.path.join(RAW, "roman_urdu.csv"), header=None,
                     names=["text", "label", "extra"], encoding="utf-8")
    SUMMARY["roman_urdu_raw_label_counts"] = df["label"].value_counts(dropna=False).to_dict()
    df = df.drop(columns=["extra"])              # 3rd column is junk (7 non-null)
    df["label"] = df["label"].replace({"Neative": "Negative"}).str.lower()
    df = basic_clean(df, "roman_urdu")
    df["language"] = "roman_urdu"
    df["source"] = "UCI #458 Roman Urdu"
    df["split"] = "unsplit"
    df["gender"] = "unknown"
    df["english_word_ratio"] = df["text"].apply(english_ratio)
    # code-switched: at least 2 English lexicon words but not entirely English
    n_eng = df["text"].apply(lambda t: sum(w in ENGLISH_WORDS for w in tokenize(t)))
    df["code_switched"] = (n_eng >= 2) & (df["english_word_ratio"] < 1.0)
    return df


def clean_tweeteval():
    frames = []
    label_map = {"0": "negative", "1": "neutral", "2": "positive"}
    for split in ["train", "val", "test"]:
        with open(os.path.join(RAW, f"tweeteval_{split}_text.txt"), encoding="utf-8") as f:
            texts = f.read().splitlines()
        with open(os.path.join(RAW, f"tweeteval_{split}_labels.txt"), encoding="utf-8") as f:
            labels = f.read().splitlines()
        assert len(texts) == len(labels), f"{split}: text/label length mismatch"
        frames.append(pd.DataFrame({"text": texts, "label": labels, "split": split}))
    df = pd.concat(frames, ignore_index=True)
    SUMMARY["tweeteval_raw_label_counts"] = df["label"].map(label_map).value_counts().to_dict()
    df["label"] = df["label"].map(label_map)
    df = basic_clean(df, "tweeteval")
    df["language"] = "english"
    df["source"] = "TweetEval sentiment"
    df["gender"] = "unknown"
    df["english_word_ratio"] = 1.0
    df["code_switched"] = False
    return df


def clean_eec():
    path = os.path.join(RAW, "Equity-Evaluation-Corpus", "Equity-Evaluation-Corpus.csv")
    df = pd.read_csv(path)
    SUMMARY["eec_raw_missing"] = df.isna().sum().to_dict()
    # Missing Race = noun phrase ("my son"), missing Emotion = neutral template
    df["Race"] = df["Race"].fillna("none")
    df["Emotion"] = df["Emotion"].fillna("none")
    df["Emotion word"] = df["Emotion word"].fillna("none")
    emotion_to_label = {"joy": "positive", "anger": "negative",
                        "sadness": "negative", "fear": "negative", "none": "neutral"}
    df = df.rename(columns={"Sentence": "text", "Gender": "gender",
                            "Race": "race", "Emotion": "emotion",
                            "Person": "person", "Template": "template"})
    df["label"] = df["emotion"].map(emotion_to_label)
    df = basic_clean(df, "eec")
    df["language"] = "english"
    df["source"] = "EEC (Kiritchenko & Mohammad 2018)"
    df["split"] = "evaluation"
    df["english_word_ratio"] = 1.0
    df["code_switched"] = False
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    columns = ["text", "label", "language", "source", "split", "gender",
               "code_switched", "n_tokens", "n_chars", "english_word_ratio"]

    ru = clean_roman_urdu()
    te = clean_tweeteval()
    eec = clean_eec()

    ru[columns].to_csv(os.path.join(OUT, "roman_urdu_clean.csv"), index=False)
    te[columns].to_csv(os.path.join(OUT, "tweeteval_clean.csv"), index=False)
    eec[columns + ["race", "emotion", "person", "template"]].to_csv(
        os.path.join(OUT, "eec_clean.csv"), index=False)

    master = pd.concat([ru[columns], te[columns], eec[columns]], ignore_index=True)
    master.to_csv(os.path.join(OUT, "sentiment_master.csv"), index=False)
    SUMMARY["master_rows"] = int(len(master))

    with open(os.path.join(OUT, "cleaning_summary.json"), "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, indent=2)
    print(json.dumps(SUMMARY, indent=2))


if __name__ == "__main__":
    main()
