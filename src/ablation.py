"""
Phase 5 - Ablation study (Track B)
Each variant removes or replaces ONE component of the final model
(frozen MiniLM embeddings + 2-hidden-layer MLP, h=512, dropout 0.2,
class-weighted cross-entropy, early stopping, trained on English + Roman Urdu)
and retrains the head with all other settings unchanged. Every variant is
trained with 3 random seeds, and the mean and standard deviation of the TEST
macro-F1 (overall, English, Roman Urdu) are reported, so that small
differences can be separated from seed noise.

Outputs:
  results/ablation_runs.csv     one row per (variant, seed)
  results/ablation_results.csv  mean / std per variant
  report/figures/ablation.png
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from baseline import load_text_splits
from config import EMB_DIM, FIG, RESULTS, SEED, ensure_dirs
from train import load_split_arrays, predict_proba, train_model

SEEDS = [42, 43, 44]


def english_only(data):
    """Remove the Roman Urdu rows from the training split (val/test unchanged)."""
    X, y, meta = data["train"]
    keep = (meta["language"] == "english").values
    out = dict(data)
    out["train"] = (X[keep], y[keep], meta[keep].reset_index(drop=True))
    return out


def tfidf_features(data):
    """Replace the MiniLM embeddings by TF-IDF (same settings as the baseline),
    reduced to the same 384 dimensions with truncated SVD and standardised.
    TF-IDF, SVD and scaler are fitted on the training split only."""
    texts = load_text_splits()
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    svd = TruncatedSVD(n_components=EMB_DIM, random_state=SEED)
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(svd.fit_transform(tfidf.fit_transform(texts["train"][0])))
    out = {}
    for name, (X, y, meta) in data.items():
        Z = Z_tr if name == "train" else scaler.transform(
            svd.transform(tfidf.transform(texts[name][0])))
        assert (y.numpy() == texts[name][1]).all()
        out[name] = (torch.tensor(Z, dtype=torch.float32), y, meta)
    return out


# name: (description, hyperparameter changes, data transformation, early-stopping patience)
VARIANTS = {
    "Full model": ("final model from Deliverable 2", {}, None, 4),
    "Linear head (no hidden layer)": ("MLP replaced by one linear layer", {"n_layers": 0}, None, 4),
    "1 hidden layer": ("second hidden layer removed", {"n_layers": 1}, None, 4),
    "No dropout": ("dropout 0.2 -> 0", {"dropout": 0.0}, None, 4),
    "No class weights": ("plain cross-entropy", {"class_weighted": False}, None, 4),
    "No early stopping": ("40 epochs, last epoch kept", {}, None, None),
    "No Roman Urdu in training": ("train on English rows only", {}, english_only, 4),
    "TF-IDF instead of MiniLM": ("TF-IDF + SVD(384) input, same MLP", {}, tfidf_features, 4),
}


def test_scores(model, data):
    X, y, meta = data["test"]
    pred = predict_proba(model, X).argmax(1)
    y = y.numpy()
    en = (meta["language"] == "english").values
    return {"test_macro_f1": f1_score(y, pred, average="macro"),
            "test_f1_english": f1_score(y[en], pred[en], average="macro"),
            "test_f1_roman_urdu": f1_score(y[~en], pred[~en], average="macro")}


def plot_ablation(summary, path):
    full = summary.loc["Full model"]
    rest = summary.drop(index="Full model")
    cols = [("test_macro_f1", "All test"), ("test_f1_english", "English"),
            ("test_f1_roman_urdu", "Roman Urdu")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    ypos = np.arange(len(rest))
    for ax, (col, title) in zip(axes, cols):
        delta = rest[col + "_mean"] - full[col + "_mean"]
        ax.barh(ypos, delta, xerr=rest[col + "_std"], capsize=3,
                color=["tab:red" if d < 0 else "tab:green" for d in delta])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{title} (full model: {full[col + '_mean']:.3f})")
        ax.set_xlabel("Change in test macro-F1 vs. full model")
        for y, d, s in zip(ypos, delta, rest[col + "_std"]):
            x = d + s if d >= 0 else d - s                 # label just past the error bar
            ax.text(x, y, f"  {d:+.3f}  ", va="center", ha="left" if d >= 0 else "right",
                    fontsize=8)
        low = min(-0.02, (delta - rest[col + "_std"]).min() * 1.45)
        high = max(0.03, (delta + rest[col + "_std"]).max() * 2.5)
        ax.set_xlim(low, high)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_yticks(ypos, rest.index)
    axes[0].invert_yaxis()
    fig.suptitle("Ablation study (mean of 3 seeds, error bar = std)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ensure_dirs()
    with open(os.path.join(RESULTS, "tuning_results.json"), encoding="utf-8") as f:
        best_params = json.load(f)["best_params"]
    base_data = load_split_arrays()

    rows = []
    for name, (desc, changes, transform, patience) in VARIANTS.items():
        data = transform(base_data) if transform else base_data
        params = {**best_params, **changes}
        for seed in SEEDS:
            start = time.time()
            model, history, best_val = train_model(params, data, patience=patience,
                                                   verbose=False, seed=seed)
            best_epoch = (len(history) if patience is None
                          else max(history, key=lambda h: h["val_macro_f1"])["epoch"])
            rows.append({"variant": name, "description": desc, "seed": seed,
                         "epochs": best_epoch, "val_macro_f1": best_val,
                         **test_scores(model, data)})
            print(f"{name:32s} seed {seed}: test macro-F1 {rows[-1]['test_macro_f1']:.4f} "
                  f"(EN {rows[-1]['test_f1_english']:.4f}, "
                  f"RU {rows[-1]['test_f1_roman_urdu']:.4f}) "
                  f"[{time.time() - start:.0f}s]")

    runs = pd.DataFrame(rows).round(4)
    runs.to_csv(os.path.join(RESULTS, "ablation_runs.csv"), index=False)

    metrics = ["val_macro_f1", "test_macro_f1", "test_f1_english", "test_f1_roman_urdu"]
    summary = runs.groupby("variant", sort=False)[metrics + ["epochs"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    full = summary.loc["Full model"]
    for m in metrics[1:]:
        summary[m + "_delta"] = summary[m + "_mean"] - full[m + "_mean"]
    summary = summary.round(4)
    summary.to_csv(os.path.join(RESULTS, "ablation_results.csv"))
    print(summary[[m + "_mean" for m in metrics[1:]] + [m + "_delta" for m in metrics[1:]]]
          .to_string())

    plot_ablation(summary, os.path.join(FIG, "ablation.png"))
    print("Saved results/ablation_results.csv and report/figures/ablation.png")


if __name__ == "__main__":
    main()
