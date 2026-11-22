"""
Phase 5 - Bias mitigation
Applies the two mitigation methods planned in Deliverables 1 and 2 that work
with a frozen encoder, and measures whether they reduce the gaps found in the
Bias Check:

1. Language re-weighting: Roman Urdu training rows are oversampled (with
   replacement) until both languages have the same number of training rows, so
   each language contributes equally to the loss. Targets the language gap.
2. Counterfactual data augmentation (CDA): every English training sentence that
   contains a gendered word (he/she, his/her, mother/father, ...) is copied with
   the gendered words swapped and the same label. The copies are embedded with
   the same frozen MiniLM encoder and added to the training set. Targets the
   gender gap. EEC sentences and names are NOT used, so the EEC stays unseen.
3. Both methods combined.

Fine-tuning the encoder itself is not done (too slow on CPU, see the report).
Every variant (and the original model) is trained with the tuned hyperparameters
and 3 seeds, and compared on test macro-F1 (overall, English, Roman Urdu) and on
the EEC gender and race score gaps (paired t-tests, as in bias_check.py).
Mean and std over the seeds are reported; the seed-42 weights are saved.

Outputs:
  models/mlp_<variant>.pt               seed-42 weights of each mitigated model
  data/processed/embeddings_cda.npy     cached embeddings of the CDA copies
  results/mitigation_runs.csv           one row per (variant, seed)
  results/mitigation_results.csv        mean / std per variant
  results/cda_examples.csv
  report/figures/mitigation.png
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from bias_check import COMPARISONS, load_eec, paired_gap
from config import (ENCODER_NAME, FIG, LABELS, MASTER_CSV, MODELS, PROC, RESULTS, SEED,
                    ensure_dirs)
from train import load_split_arrays, load_trained_model, predict_proba, train_model

CDA_NPY = os.path.join(PROC, "embeddings_cda.npy")
CDA_CSV = os.path.join(PROC, "cda_rows.csv")
SEEDS = [42, 43, 44]

# (male, female) word pairs; swapping is done in both directions
GENDER_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "her"), ("his", "hers"), ("himself", "herself"),
    ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"),
    ("father", "mother"), ("fathers", "mothers"), ("dad", "mom"), ("dad", "mum"),
    ("son", "daughter"), ("sons", "daughters"), ("brother", "sister"),
    ("brothers", "sisters"), ("husband", "wife"), ("husbands", "wives"),
    ("boyfriend", "girlfriend"), ("uncle", "aunt"), ("nephew", "niece"),
    ("king", "queen"), ("prince", "princess"), ("mr", "mrs"), ("sir", "madam"),
    ("gentleman", "lady"), ("gentlemen", "ladies"), ("male", "female"),
    ("grandfather", "grandmother"), ("grandpa", "grandma"), ("actor", "actress"),
    ("groom", "bride"),
]
SWAP = {}
for _m, _f in GENDER_PAIRS:
    SWAP.setdefault(_m, _f)
    SWAP.setdefault(_f, _m)
GENDER_RE = re.compile(r"\b(" + "|".join(sorted(SWAP, key=len, reverse=True)) + r")\b",
                       re.IGNORECASE)
# "her" followed by one of these words is an object pronoun ("told her that") -> "him"
OBJECT_NEXT = {"a", "an", "the", "that", "this", "and", "or", "but", "so", "to", "for",
               "in", "on", "at", "of", "with", "about", "from", "is", "was", "too",
               "again", "back", "up", "out", "off", "now", "today", "tonight", "more"}


def _swap_word(match):
    word = match.group(0)
    low = word.lower()
    if low == "her":   # "her" is "his" before a noun ("her phone"), otherwise "him"
        nxt = re.match(r"\s+([A-Za-z]+)", match.string[match.end():])
        new = "his" if nxt and nxt.group(1).lower() not in OBJECT_NEXT else "him"
    else:
        new = SWAP[low]
    if word.isupper() and len(word) > 1:
        return new.upper()
    return new.capitalize() if word[0].isupper() else new


def swap_gender(text):
    return GENDER_RE.sub(_swap_word, text)


def cda_embeddings(meta_train):
    """Gender-swapped copies of the English training rows that contain a gendered
    word, with their MiniLM embeddings (computed once and cached)."""
    texts = pd.read_csv(MASTER_CSV, usecols=["text"])["text"]
    en = meta_train[meta_train["language"] == "english"]
    original = texts.loc[en["row_id"]].astype(str)
    swapped = original.map(swap_gender)
    changed = swapped != original
    rows = pd.DataFrame({"train_pos": en.index[changed.values],
                         "row_id": en["row_id"].values[changed.values],
                         "original": original[changed].values,
                         "swapped": swapped[changed].values})

    if os.path.exists(CDA_NPY) and os.path.exists(CDA_CSV):
        cached = pd.read_csv(CDA_CSV)
        if cached["row_id"].tolist() == rows["row_id"].tolist():
            return rows, np.load(CDA_NPY)
    from sentence_transformers import SentenceTransformer
    print(f"Embedding {len(rows)} gender-swapped sentences with {ENCODER_NAME} ...")
    encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
    emb = encoder.encode(rows["swapped"].tolist(), batch_size=128, show_progress_bar=True,
                         convert_to_numpy=True).astype(np.float32)
    np.save(CDA_NPY, emb)
    rows[["row_id", "swapped"]].to_csv(CDA_CSV, index=False)
    return rows, emb


def add_cda(data, rows, emb):
    """Training split + the gender-swapped copies (same labels as the originals)."""
    X, y, meta = data["train"]
    pos = rows["train_pos"].values
    out = dict(data)
    out["train"] = (torch.cat([X, torch.tensor(emb)]), torch.cat([y, y[pos]]),
                    pd.concat([meta, meta.iloc[pos]], ignore_index=True))
    return out


def balance_languages(data, seed=SEED):
    """Oversample Roman Urdu training rows until both languages have the same count."""
    X, y, meta = data["train"]
    en = (meta["language"] == "english").values
    ru_idx = np.where(~en)[0]
    extra = np.random.default_rng(seed).choice(ru_idx, en.sum() - len(ru_idx), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    out = dict(data)
    out["train"] = (X[idx], y[idx], meta.iloc[idx].reset_index(drop=True))
    return out


def evaluate_variant(model, data, eec):
    X, y, meta = data["test"]
    y = y.numpy()
    pred = predict_proba(model, X).argmax(1)
    en = (meta["language"] == "english").values
    res = {"test_accuracy": accuracy_score(y, pred),
           "test_macro_f1": f1_score(y, pred, average="macro"),
           "test_f1_english": f1_score(y[en], pred[en], average="macro"),
           "test_f1_roman_urdu": f1_score(y[~en], pred[~en], average="macro")}
    res["language_gap"] = res["test_f1_english"] - res["test_f1_roman_urdu"]

    Xb, yb, _ = data["bias"]
    prob = predict_proba(model, Xb)
    df = eec.copy()
    df["score"] = prob[:, LABELS.index("positive")] - prob[:, LABELS.index("negative")]
    res["eec_accuracy"] = accuracy_score(yb.numpy(), prob.argmax(1))
    for name, (col, a, b, rows) in COMPARISONS.items():
        part = df[df["Race"].notna()] if rows == "names" else df
        summary, _ = paired_gap(part, col, a, b)
        key = name.split()[0]                      # "gender" / "race"
        res[f"{key}_gap"] = summary["mean_gap"]
        res[f"{key}_p_value"] = summary["p_value"]
    # gender gap separately for name sentences and for noun-phrase sentences
    # ("she", "my mother"); CDA swaps only noun phrases / pronouns, not names
    for key, part in [("names", df[df["Race"].notna()]), ("nouns", df[df["Race"].isna()])]:
        res[f"gender_gap_{key}"] = paired_gap(part, "Gender", "female", "male")[0]["mean_gap"]
    return res


def plot_mitigation(summary, path):
    names = summary.index.tolist()
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    w = 0.38
    for off, col, lbl in [(-w / 2, "test_f1_english", "English"),
                          (w / 2, "test_f1_roman_urdu", "Roman Urdu")]:
        bars = axes[0].bar(x + off, summary[col + "_mean"], w, yerr=summary[col + "_std"],
                           capsize=3, label=lbl)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=7, padding=4)
    axes[0].set_ylim(0.4, 0.75)
    axes[0].set_ylabel("Test macro-F1")
    axes[0].set_title("Language gap (test split)")
    axes[0].legend(loc="upper right", fontsize=8)

    w = 0.26
    for off, col, lbl in [(-w, "gender_gap", "Gender, all EEC rows"),
                          (0, "gender_gap_nouns", "Gender, noun-phrase rows"),
                          (w, "race_gap", "Race (names)")]:
        bars = axes[1].bar(x + off, summary[col + "_mean"], w, yerr=summary[col + "_std"],
                           capsize=3, label=lbl)
        axes[1].bar_label(bars, fmt="%+.3f", fontsize=6.5, padding=4)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Mean gap in P(pos) - P(neg)")
    axes[1].set_title("EEC score gaps (female - male, Afr.-American - European)")
    axes[1].legend(loc="lower left", fontsize=8)

    for ax in axes:
        ax.set_xticks(x, [n.replace(" + ", " +\n").replace(" (", "\n(") for n in names],
                      fontsize=8)
    fig.suptitle(f"Mitigation results (mean of {len(SEEDS)} seeds, error bar = std)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ensure_dirs()
    with open(os.path.join(RESULTS, "tuning_results.json"), encoding="utf-8") as f:
        params = json.load(f)["best_params"]
    data = load_split_arrays()
    eec = load_eec(data["bias"][2])

    rows, emb = cda_embeddings(data["train"][2])
    print(f"CDA: {len(rows)} gender-swapped copies added to the training split")
    rows[["original", "swapped"]].head(15).to_csv(os.path.join(RESULTS, "cda_examples.csv"),
                                                  index=False)
    cda_data = add_cda(data, rows, emb)

    # name: (file for the seed-42 weights, training data for a given seed)
    variants = {
        "Original (Deliverable 2)": (None, lambda s: data),
        "Language re-weighting": ("mlp_lang_balanced.pt", lambda s: balance_languages(data, s)),
        "Gender CDA": ("mlp_gender_cda.pt", lambda s: cda_data),
        "Re-weighting + CDA": ("mlp_lang_balanced_cda.pt",
                               lambda s: balance_languages(cda_data, s)),
    }

    runs = []
    for name, (filename, make_data) in variants.items():
        for seed in SEEDS:
            train_data = make_data(seed)
            if filename is None and seed == SEED:
                model = load_trained_model()       # identical to retraining with seed 42
            else:
                model = train_model(params, train_data, verbose=False, seed=seed)[0]
            if filename and seed == SEED:
                torch.save({"state_dict": model.state_dict(), "params": params},
                           os.path.join(MODELS, filename))
            runs.append({"variant": name, "seed": seed,
                         "train_rows": len(train_data["train"][1]),
                         **evaluate_variant(model, data, eec)})
            print(f"{name:26s} seed {seed}: RU F1 {runs[-1]['test_f1_roman_urdu']:.4f}, "
                  f"gap {runs[-1]['language_gap']:.4f}, gender gap {runs[-1]['gender_gap']:+.4f}")

    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(RESULTS, "mitigation_runs.csv"), index=False)

    metrics = [c for c in runs.columns if c not in ("variant", "seed", "train_rows")
               and not c.endswith("p_value")]
    summary = runs.groupby("variant", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.round(4)
    summary.insert(0, "train_rows", runs.groupby("variant", sort=False)["train_rows"].first())
    for key in ("gender", "race"):                 # in how many seeds is the gap significant
        summary[f"{key}_significant_runs"] = (runs[f"{key}_p_value"] < 0.05).groupby(
            runs["variant"], sort=False).sum().astype(str) + f"/{len(SEEDS)}"
    summary.to_csv(os.path.join(RESULTS, "mitigation_results.csv"))
    print(summary.to_string())
    plot_mitigation(summary, os.path.join(FIG, "mitigation.png"))
    print("Saved results/mitigation_results.csv and report/figures/mitigation.png")


if __name__ == "__main__":
    main()
