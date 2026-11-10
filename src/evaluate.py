"""
Phase 4 - Metric analysis and error analysis on the held-out TEST split
Evaluates both models (TF-IDF + Logistic Regression baseline, MiniLM + MLP)
with accuracy, macro precision / recall / F1 and one-vs-rest macro AUC-ROC,
per-class scores, per-subgroup scores (language, code-switching) and an error
analysis (confusion pairs and the most confident mistakes).

Outputs:
  results/test_metrics.json, results/subgroup_metrics.csv, results/error_examples.csv
  report/figures/confusion_matrices.png, roc_curves.png, subgroup_f1.png
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from baseline import load_text_splits
from config import FIG, LABELS, MASTER_CSV, MODELS, RESULTS, ensure_dirs
from train import load_trained_model, load_split_arrays, predict_proba

MODEL_NAMES = ["TF-IDF + LogReg", "MiniLM + MLP"]


def get_probs(split):
    """Class probabilities of both models on one split, plus labels and metadata."""
    emb_data = load_split_arrays()
    X_emb, y, meta = emb_data[split]
    texts, y_text = load_text_splits()[split]
    assert (y.numpy() == y_text).all()

    baseline = joblib.load(os.path.join(MODELS, "baseline_tfidf_lr.joblib"))
    mlp = load_trained_model()
    probs = {"TF-IDF + LogReg": baseline.predict_proba(texts),
             "MiniLM + MLP": predict_proba(mlp, X_emb)}
    return probs, y.numpy(), meta


def compute_metrics(y, prob):
    pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    m = {"n": int(len(y)), "accuracy": accuracy_score(y, pred),
         "macro_precision": p, "macro_recall": r, "macro_f1": f1}
    m["auc_roc_macro_ovr"] = (roc_auc_score(y, prob, multi_class="ovr", average="macro")
                              if len(np.unique(y)) == len(LABELS) else float("nan"))
    return {k: (round(float(v), 4) if k != "n" else v) for k, v in m.items()}


def per_class(y, prob):
    p, r, f1, s = precision_recall_fscore_support(y, prob.argmax(1), labels=[0, 1, 2],
                                                  zero_division=0)
    return {LABELS[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                        "f1": round(float(f1[i]), 4), "support": int(s[i])} for i in range(3)}


def subgroup_table(y, probs, meta):
    groups = {
        "All test": np.ones(len(y), bool),
        "English (TweetEval)": (meta["language"] == "english").values,
        "Roman Urdu (all)": (meta["language"] == "roman_urdu").values,
        "Roman Urdu, code-switched": ((meta["language"] == "roman_urdu")
                                      & meta["code_switched"]).values,
        "Roman Urdu, not code-switched": ((meta["language"] == "roman_urdu")
                                          & ~meta["code_switched"]).values,
    }
    rows = []
    for model, prob in probs.items():
        for g, mask in groups.items():
            rows.append({"model": model, "group": g, **compute_metrics(y[mask], prob[mask])})
    return pd.DataFrame(rows)


def plot_confusion(y, probs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, prob) in zip(axes, probs.items()):
        cm = confusion_matrix(y, prob.argmax(1), normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS, cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Row-normalised confusion matrices (test split)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y, probs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, prob) in zip(axes, probs.items()):
        for i, lbl in enumerate(LABELS):
            fpr, tpr, _ = roc_curve(y == i, prob[:, i])
            ax.plot(fpr, tpr, label=f"{lbl} (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(f"{name}: one-vs-rest ROC")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_subgroups(table, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=table, x="group", y="macro_f1", hue="model", ax=ax)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("")
    ax.set_title("Test macro-F1 by language subgroup")
    ax.set_xticks(ax.get_xticks(), [t.get_text().replace(", ", ",\n")
                                    for t in ax.get_xticklabels()], fontsize=8)
    ax.set_ylim(0, 1)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_analysis(y, probs, meta):
    """Most confident mistakes of each model and counts of each confusion pair."""
    texts = pd.read_csv(MASTER_CSV, usecols=["text"]).loc[meta["row_id"], "text"].values
    examples, pairs = [], {}
    for name, prob in probs.items():
        pred = prob.argmax(1)
        wrong = np.where(pred != y)[0]
        pair_counts = pd.Series([f"{LABELS[y[i]]} -> {LABELS[pred[i]]}" for i in wrong])
        pairs[name] = pair_counts.value_counts().to_dict()
        top = wrong[np.argsort(-prob[wrong].max(1))][:15]
        for i in top:
            examples.append({"model": name, "language": meta["language"].iat[i],
                             "true": LABELS[y[i]], "pred": LABELS[pred[i]],
                             "confidence": round(float(prob[i].max()), 3), "text": texts[i]})
    return pd.DataFrame(examples), pairs


def main():
    ensure_dirs()
    probs, y, meta = get_probs("test")

    results = {}
    for name, prob in probs.items():
        results[name] = {"overall": compute_metrics(y, prob), "per_class": per_class(y, prob)}
        print(name, results[name]["overall"])

    table = subgroup_table(y, probs, meta)
    table.to_csv(os.path.join(RESULTS, "subgroup_metrics.csv"), index=False)
    print(table.to_string(index=False))

    examples, pairs = error_analysis(y, probs, meta)
    examples.to_csv(os.path.join(RESULTS, "error_examples.csv"), index=False)
    for name in probs:
        results[name]["confusion_pairs"] = pairs[name]
    with open(os.path.join(RESULTS, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_confusion(y, probs, os.path.join(FIG, "confusion_matrices.png"))
    plot_roc(y, probs, os.path.join(FIG, "roc_curves.png"))
    plot_subgroups(table, os.path.join(FIG, "subgroup_f1.png"))
    print("Saved metrics to results/ and figures to report/figures/")


if __name__ == "__main__":
    main()
