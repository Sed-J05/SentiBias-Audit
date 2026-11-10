"""
Phase 4 - Bias Check on the Equity Evaluation Corpus (EEC)
EEC sentences come from fixed templates that differ only in the person
(female vs male names / noun phrases, African-American vs European names).
Neither model saw EEC during training, so any systematic score difference
between the groups is bias learned from the training data or the encoder.

For every sentence the sentiment score is  s = P(positive) - P(negative).
Following Kiritchenko & Mohammad (2018), sentences are grouped by
(template, emotion word); within each group the mean score of the female (or
African-American) sentences is compared with the male (or European) ones,
and a paired t-test over all groups tests whether the gap is significant.

Outputs:
  results/bias_check.json, results/bias_by_emotion.csv
  report/figures/bias_check.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score

from config import FIG, LABELS, MASTER_CSV, RESULTS, ROOT, ensure_dirs
from evaluate import get_probs

EEC_RAW = os.path.join(ROOT, "data", "raw", "Equity-Evaluation-Corpus",
                       "Equity-Evaluation-Corpus.csv")
COMPARISONS = {
    # name: (column, group A, group B, row filter)
    "gender (female - male)": ("Gender", "female", "male", None),
    "race (African-American - European)": ("Race", "African-American", "European", "names"),
}


def load_eec(meta):
    """EEC metadata aligned with the bias split (Phase 2 kept all 8,640 rows in order)."""
    eec = pd.read_csv(EEC_RAW)
    texts = pd.read_csv(MASTER_CSV, usecols=["text"]).loc[meta["row_id"], "text"].values
    assert len(eec) == len(texts) and (eec["Sentence"].str.strip().values == texts).all()
    eec["Emotion"] = eec["Emotion"].fillna("none")
    eec["Emotion word"] = eec["Emotion word"].fillna("none")
    return eec


def paired_gap(df, col, a, b):
    """Mean score per (template, emotion word) for group a and b, then a paired t-test."""
    means = (df[df[col].isin([a, b])]
             .groupby(["Template", "Emotion word", "Emotion", col])["score"].mean()
             .unstack(col).dropna())
    diff = means[a] - means[b]
    t, p = ttest_rel(means[a], means[b])
    summary = {"n_pairs": int(len(diff)), "mean_score_" + a: round(float(means[a].mean()), 4),
               "mean_score_" + b: round(float(means[b].mean()), 4),
               "mean_gap": round(float(diff.mean()), 4),
               "mean_abs_gap": round(float(diff.abs().mean()), 4),
               "share_pairs_a_higher": round(float((diff > 0).mean()), 4),
               "t_statistic": round(float(t), 3), "p_value": float(f"{p:.3g}"),
               "significant_at_0.05": bool(p < 0.05)}
    by_emotion = diff.groupby(level="Emotion").mean()
    return summary, by_emotion


def group_performance(df, col, groups):
    out = {}
    for g in groups:
        part = df[df[col] == g]
        out[g] = {"n": int(len(part)),
                  "accuracy": round(accuracy_score(part["y"], part["pred"]), 4),
                  "macro_f1": round(f1_score(part["y"], part["pred"], average="macro"), 4),
                  "share_pred_positive": round(float((part["pred"] == 2).mean()), 4),
                  "share_pred_negative": round(float((part["pred"] == 0).mean()), 4)}
    return out


def main():
    ensure_dirs()
    probs, y, meta = get_probs("bias")
    eec = load_eec(meta)

    results, emotion_rows = {}, []
    for model, prob in probs.items():
        df = eec.copy()
        df["score"] = prob[:, LABELS.index("positive")] - prob[:, LABELS.index("negative")]
        df["y"], df["pred"] = y, prob.argmax(1)
        res = {"eec_overall": {"accuracy": round(accuracy_score(y, df["pred"]), 4),
                               "macro_f1": round(f1_score(y, df["pred"], average="macro"), 4)}}
        for name, (col, a, b, rows) in COMPARISONS.items():
            part = df[df["Race"].notna()] if rows == "names" else df
            summary, by_emotion = paired_gap(part, col, a, b)
            summary["performance"] = group_performance(part, col, [a, b])
            res[name] = summary
            for emo, gap in by_emotion.items():
                emotion_rows.append({"model": model, "comparison": name,
                                     "emotion": emo, "mean_gap": round(float(gap), 4)})
        results[model] = res
        print(model, json.dumps(res, indent=2))

    with open(os.path.join(RESULTS, "bias_check.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    emo = pd.DataFrame(emotion_rows)
    emo.to_csv(os.path.join(RESULTS, "bias_by_emotion.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, COMPARISONS):
        sns.barplot(data=emo[emo["comparison"] == name], x="emotion", y="mean_gap",
                    hue="model", ax=ax, order=["anger", "fear", "joy", "sadness", "none"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title(f"Score gap, {name}")
        ax.set_xlabel("EEC emotion of the template")
        ax.set_ylabel("Mean gap in P(pos) - P(neg)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bias_check.png"), dpi=150)
    plt.close(fig)
    print("Saved results/bias_check.json and report/figures/bias_check.png")


if __name__ == "__main__":
    main()
