"""
Phase 5 - Inference and sample predictions
Loads every trained model and predicts the sentiment of new text.

Usage:
  python src/predict.py "Yeh phone bohat acha hai" "Worst service ever"
      prints the prediction of every model for each text
  python src/predict.py
      runs the fixed demo set below (English, Roman Urdu, code-switched, negation,
      sarcasm and a gender-swapped pair) and saves
      results/sample_predictions.csv and report/figures/sample_predictions.png

The baseline needs the same text normalisation as Phase 2 (feature_eda.py); the
MLP models need the frozen MiniLM embedding of the raw text (as in embed.py).
"""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import ENCODER_NAME, FIG, LABELS, MODELS, RESULTS, ROOT, ensure_dirs
from train import load_trained_model, predict_proba

sys.path.insert(0, os.path.join(ROOT, "data_scripts"))
from feature_eda import normalize_text  # noqa: E402  (same normalisation as Phase 2)

MLP_FILES = {"MiniLM + MLP": "sentiment_mlp.pt",
             "MLP + re-weighting": "mlp_lang_balanced.pt",
             "MLP + CDA": "mlp_gender_cda.pt",
             "MLP + re-weighting + CDA": "mlp_lang_balanced_cda.pt"}

# (text, expected label, type) - written by us, not taken from any dataset
DEMO = [
    ("I absolutely love the new update, great job!", "positive", "English"),
    ("Worst customer service ever, never ordering again.", "negative", "English"),
    ("The meeting has been moved to 5 pm on Friday.", "neutral", "English"),
    ("Great, my flight got cancelled again. Just perfect.", "negative", "English, sarcasm"),
    ("Yeh phone bohat acha hai, battery bhi zabardast hai", "positive", "Roman Urdu"),
    ("Khana bilkul bekar tha aur service bhi slow thi", "negative", "Roman Urdu"),
    ("Main kal Lahore ja raha hoon", "neutral", "Roman Urdu"),
    ("Yeh drama acha nahi tha", "negative", "Roman Urdu, negation"),
    ("Movie zabardast thi, I loved it", "positive", "Code-switched"),
    ("Delivery itni late thi, totally disappointed", "negative", "Code-switched"),
    ("Her presentation is scheduled for today.", "neutral", "Gender pair (female)"),
    ("His presentation is scheduled for today.", "neutral", "Gender pair (male)"),
]


def load_models():
    models = {"TF-IDF + LogReg": joblib.load(os.path.join(MODELS, "baseline_tfidf_lr.joblib"))}
    for name, filename in MLP_FILES.items():
        if os.path.exists(os.path.join(MODELS, filename)):
            models[name] = load_trained_model(filename)
    return models


def predict(texts, models, encoder=None):
    """Return {model name: (n, 3) probability array} for a list of raw texts."""
    out = {}
    if "TF-IDF + LogReg" in models:
        out["TF-IDF + LogReg"] = models["TF-IDF + LogReg"].predict_proba(
            [normalize_text(t) for t in texts])
    mlps = {n: m for n, m in models.items() if n != "TF-IDF + LogReg"}
    if mlps:
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
        X = torch.tensor(encoder.encode(texts, convert_to_numpy=True).astype(np.float32))
        for name, model in mlps.items():
            out[name] = predict_proba(model, X)
    return out


def to_table(texts, probs):
    rows = []
    for i, t in enumerate(texts):
        row = {"text": t}
        for name, p in probs.items():
            row[f"{name} | pred"] = LABELS[p[i].argmax()]
            row[f"{name} | conf"] = round(float(p[i].max()), 3)
            row[f"{name} | score"] = round(float(p[i][2] - p[i][0]), 3)   # P(pos) - P(neg)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_samples(demo, probs, path):
    names = list(probs)
    score = np.stack([p[:, 2] - p[:, 0] for p in probs.values()], axis=1)
    fig, ax = plt.subplots(figsize=(3.0 + 1.9 * len(names), 0.48 * len(demo) + 1.6))
    im = ax.imshow(score, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(demo)):
        for j, p in enumerate(probs.values()):
            pred = LABELS[p[i].argmax()]
            mark = "" if pred == demo[i][1] else " x"
            ax.text(j, i, f"{pred} {p[i].max():.2f}{mark}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(score[i, j]) > 0.6 else "black")
    ax.set_xticks(range(len(names)), [n.replace(" + ", " +\n", 1) for n in names], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(demo)),
                  [f"{t[:46] + '...' if len(t) > 49 else t}  [{e}]" for t, e, _ in demo],
                  fontsize=7.5)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("P(positive) - P(negative)", fontsize=8)
    ax.set_title("Sample predictions (label + confidence; x = differs from expected "
                 "label in [ ])", fontsize=9, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    models = load_models()
    if len(sys.argv) > 1:                          # predict user-supplied texts
        texts = sys.argv[1:]
        table = to_table(texts, predict(texts, models))
        for _, row in table.iterrows():
            print(f"\n{row['text']}")
            for name in models:
                print(f"  {name:26s} {row[f'{name} | pred']:8s} "
                      f"(confidence {row[f'{name} | conf']:.2f}, "
                      f"score {row[f'{name} | score']:+.2f})")
        return

    ensure_dirs()
    texts = [t for t, _, _ in DEMO]
    probs = predict(texts, models)
    table = to_table(texts, probs)
    table.insert(1, "expected", [e for _, e, _ in DEMO])
    table.insert(2, "type", [k for _, _, k in DEMO])
    table.to_csv(os.path.join(RESULTS, "sample_predictions.csv"), index=False)
    print(table[["text", "expected"] + [c for c in table.columns if c.endswith("| pred")]]
          .to_string(index=False))
    plot_samples(DEMO, probs, os.path.join(FIG, "sample_predictions.png"))
    print("Saved results/sample_predictions.csv and report/figures/sample_predictions.png")


if __name__ == "__main__":
    main()
