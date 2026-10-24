"""
Phase 2 - Feature Strategy and Exploratory Data Analysis
Reads data/processed/sentiment_master.csv and
  * normalises text (lowercase, URL/mention/number placeholders),
  * encodes variables (label encoding for the target, one-hot for language
    and gender, TF-IDF for the text),
  * scales numeric features (StandardScaler and MinMaxScaler),
  * saves the feature files to data/processed/ and EDA charts to report/figures/.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

BASE = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "report", "figures")

sns.set_theme(style="whitegrid")


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " @user ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s@']", " ", text)      # drop punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_features(df):
    # --- Normalisation of the text itself
    df["text_norm"] = df["text"].apply(normalize_text)

    # --- Encoding
    le = LabelEncoder()                           # negative=0, neutral=1, positive=2
    df["label_enc"] = le.fit_transform(df["label"])
    onehot = pd.get_dummies(df[["language", "gender"]], dtype=int)
    df["code_switched"] = df["code_switched"].astype(int)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_tfidf = tfidf.fit_transform(df["text_norm"])
    save_npz(os.path.join(PROC, "features_tfidf.npz"), X_tfidf)

    # --- Scaling of numeric features
    numeric = df[["n_tokens", "n_chars", "english_word_ratio"]]
    std = pd.DataFrame(StandardScaler().fit_transform(numeric),
                       columns=[c + "_std" for c in numeric.columns])
    mm = pd.DataFrame(MinMaxScaler().fit_transform(numeric),
                      columns=[c + "_minmax" for c in numeric.columns])

    features = pd.concat([df[["label_enc", "code_switched"]], onehot, std, mm], axis=1)
    features.to_csv(os.path.join(PROC, "features_tabular.csv"), index=False)
    df.to_csv(os.path.join(PROC, "sentiment_master_normalized.csv"), index=False)

    info = {
        "label_classes": list(le.classes_),
        "onehot_columns": list(onehot.columns),
        "tfidf_shape": list(X_tfidf.shape),
        "tabular_columns": list(features.columns),
    }
    with open(os.path.join(PROC, "feature_summary.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    return df, features


def make_charts(df, features):
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(PROC, "cleaning_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    # 1. Raw class distribution per dataset (before cleaning)
    raw = []
    for name, key in [("Roman Urdu", "roman_urdu_raw_label_counts"),
                      ("TweetEval", "tweeteval_raw_label_counts")]:
        for lbl, n in summary[key].items():
            raw.append({"dataset": name, "label": str(lbl), "count": n})
    raw = pd.DataFrame(raw)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=raw, x="label", y="count", hue="dataset", ax=ax)
    ax.set_title("Raw class distribution (before cleaning)")
    ax.set_xlabel("Raw label value")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "raw_class_distribution.png"), dpi=150)

    # 2. Cleaned class distribution per source
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="source", hue="label", ax=ax,
                  hue_order=["negative", "neutral", "positive"])
    ax.set_title("Class distribution after cleaning")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "clean_class_distribution.png"), dpi=150)

    # 3. Text length distribution by language
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="n_tokens", hue="language", bins=40,
                 element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title("Token count per sample by language (after outlier removal)")
    ax.set_xlabel("Tokens")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "text_length_distribution.png"), dpi=150)

    # 4. Code-switching in Roman Urdu by label
    ru = df[df["language"] == "roman_urdu"]
    cs = ru.groupby("label")["code_switched"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=cs, x="label", y="code_switched", ax=ax)
    ax.set_ylabel("Share of code-switched samples")
    ax.set_title("Code-switched (Roman Urdu + English) share by label")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "code_switching_by_label.png"), dpi=150)

    # 5. EEC gender x label balance
    eec = df[df["source"].str.startswith("EEC")]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=eec, x="label", hue="gender", ax=ax,
                  order=["negative", "neutral", "positive"])
    ax.set_title("EEC: gender-coded sentences per sentiment label")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eec_gender_balance.png"), dpi=150)

    # 6. Feature correlation heatmap
    corr_cols = ["label_enc", "code_switched", "n_tokens_std", "n_chars_std",
                 "english_word_ratio_std", "language_roman_urdu", "gender_female"]
    corr = features[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "feature_correlation.png"), dpi=150)
    print("Saved 6 charts to", os.path.abspath(FIG))


def main():
    df = pd.read_csv(os.path.join(PROC, "sentiment_master.csv"))
    df, features = build_features(df)
    make_charts(df, features)


if __name__ == "__main__":
    main()
