"""
Phase 5 - Final comparative analysis and failure analysis (Track B)

1. Comparison: every model is scored on the held-out TEST split:
     majority-class baseline, TF-IDF + LogReg baseline, MiniLM + MLP (ours) and
     the mitigated MLPs from mitigate.py (if their weights exist).
   Macro-F1 gets a 95% bootstrap confidence interval (1,000 resamples) for all
   test rows, English rows and Roman Urdu rows.
2. Significance: for pairs of models, an exact McNemar test on per-sample
   correctness and a paired bootstrap test on the macro-F1 difference.
3. Failure analysis: error rates of both main models on slices of the test set
   (language x text length, negation words, code-switching), an estimate of the
   label noise in the raw Roman Urdu file (duplicate texts with conflicting
   labels) and the MiniLM tokenizer's sub-word pieces per word in each language,
   to explain WHY samples fail.

Outputs:
  results/final_comparison.csv, results/significance.csv,
  results/failure_slices.csv, results/failure_stats.json
  report/figures/final_comparison.png, report/figures/failure_slices.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from config import (ENCODER_NAME, FIG, MASTER_CSV, MODELS, RESULTS, ROOT, SEED,
                    ensure_dirs)
from evaluate import get_probs
from train import load_split_arrays, load_trained_model, predict_proba

N_BOOT = 1000
OURS, BASE = "MiniLM + MLP", "TF-IDF + LogReg"
MITIGATED = {"MLP + language re-weighting": "mlp_lang_balanced.pt",
             "MLP + gender CDA": "mlp_gender_cda.pt",
             "MLP + re-weighting + CDA": "mlp_lang_balanced_cda.pt"}
NEGATION = {
    "english": r"\b(?:not|no|never|nothing|nobody|cannot|cant|dont|doesnt|didnt|isnt|wasnt|"
               r"wont|aint)\b|n't\b",
    "roman_urdu": r"\b(?:nahi|nahin|nai|nhi|nahe|mat|na)\b",
}


def macro_f1(y, pred):
    """Fast macro-F1 from the confusion matrix. Gives the same value as sklearn's
    f1_score(average="macro", zero_division=0): the mean over the classes that occur
    in y or pred."""
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.nan_to_num(tp / cm.sum(0))
        rec = np.nan_to_num(tp / cm.sum(1))
        f1 = np.nan_to_num(2 * prec * rec / (prec + rec))
    present = (cm.sum(0) + cm.sum(1)) > 0
    return f1[present].mean()


def all_test_probs():
    probs, y, meta = get_probs("test")
    probs = {BASE: probs["TF-IDF + LogReg"], OURS: probs["MiniLM + MLP"]}
    y_train = load_split_arrays()["train"][1].numpy()
    prior = np.bincount(y_train, minlength=3) / len(y_train)
    probs = {"Majority class": np.tile(prior, (len(y), 1)), **probs}
    X_test = load_split_arrays()["test"][0]
    for name, filename in MITIGATED.items():
        if os.path.exists(os.path.join(MODELS, filename)):
            probs[name] = predict_proba(load_trained_model(filename), X_test)
        else:
            print(f"models/{filename} not found - run mitigate.py first; skipping {name}")
    return probs, y, meta


def comparison_table(probs, y, groups, boot_idx):
    rows = []
    for name, prob in probs.items():
        pred = prob.argmax(1)
        for g, mask in groups.items():
            sel = np.where(mask)[0]
            boot = [macro_f1(y[sel[b]], pred[sel[b]]) for b in boot_idx[g]]
            auc = roc_auc_score(y[sel], prob[sel], multi_class="ovr", average="macro")
            rows.append({"model": name, "group": g, "n": len(sel),
                         "accuracy": (pred[sel] == y[sel]).mean(),
                         "macro_f1": macro_f1(y[sel], pred[sel]),
                         "f1_ci_low": np.percentile(boot, 2.5),
                         "f1_ci_high": np.percentile(boot, 97.5),
                         "auc_roc": auc})
    return pd.DataFrame(rows).round(4)


def significance(probs, y, groups, boot_idx, pairs):
    rows = []
    for a, b in pairs:
        if a not in probs or b not in probs:
            continue
        pa, pb = probs[a].argmax(1), probs[b].argmax(1)
        for g, mask in groups.items():
            sel = np.where(mask)[0]
            ca, cb = pa[sel] == y[sel], pb[sel] == y[sel]
            only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
            mcnemar_p = binomtest(only_a, only_a + only_b, 0.5).pvalue
            diff = np.array([macro_f1(y[sel[i]], pa[sel[i]]) - macro_f1(y[sel[i]], pb[sel[i]])
                             for i in boot_idx[g]])
            rows.append({"model_a": a, "model_b": b, "group": g,
                         "f1_diff": round(macro_f1(y[sel], pa[sel]) - macro_f1(y[sel], pb[sel]), 4),
                         "diff_ci_low": round(np.percentile(diff, 2.5), 4),
                         "diff_ci_high": round(np.percentile(diff, 97.5), 4),
                         "bootstrap_p": round(min(1.0, 2 * min((diff <= 0).mean(),
                                                               (diff >= 0).mean())), 4),
                         "only_a_correct": only_a, "only_b_correct": only_b,
                         "mcnemar_p": float(f"{mcnemar_p:.3g}")})
    return pd.DataFrame(rows)


def failure_slices(probs, y, meta):
    texts = pd.read_csv(MASTER_CSV, usecols=["text_norm"]).loc[meta["row_id"], "text_norm"]
    texts = texts.fillna("").reset_index(drop=True)
    lang = meta["language"].values
    neg = np.zeros(len(texts), bool)
    for l, pattern in NEGATION.items():
        neg[lang == l] = texts[lang == l].str.contains(pattern, regex=True).values
    length = pd.cut(meta["n_tokens"], [0, 5, 15, 1000],
                    labels=["1-5 tokens", "6-15 tokens", ">15 tokens"]).astype(str).values
    slices = {}
    for l, lname in [("english", "English"), ("roman_urdu", "Roman Urdu")]:
        is_l = lang == l
        for b in ["1-5 tokens", "6-15 tokens", ">15 tokens"]:
            slices[f"{lname}, {b}"] = is_l & (length == b)
        slices[f"{lname}, with negation"] = is_l & neg
        slices[f"{lname}, no negation"] = is_l & ~neg
    cs = meta["code_switched"].values.astype(bool)
    slices["Roman Urdu, code-switched"] = (lang == "roman_urdu") & cs
    slices["Roman Urdu, not code-switched"] = (lang == "roman_urdu") & ~cs

    rows = []
    for name in [BASE, OURS]:
        pred = probs[name].argmax(1)
        for s, mask in slices.items():
            rows.append({"model": name, "slice": s, "n": int(mask.sum()),
                         "error_rate": round(float((pred[mask] != y[mask]).mean()), 4),
                         "macro_f1": round(macro_f1(y[mask], pred[mask]), 4)})
    return pd.DataFrame(rows)


def label_noise():
    """Duplicate texts in the raw Roman Urdu file that carry different labels."""
    raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "roman_urdu.csv"), header=None,
                      names=["text", "label", "extra"], encoding="utf-8")
    raw = raw.dropna(subset=["text", "label"])
    raw["text"] = raw["text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    raw["label"] = raw["label"].replace({"Neative": "Negative"}).str.lower()
    dup = raw[raw.duplicated("text", keep=False)]
    n_labels = dup.groupby("text")["label"].nunique()
    return {"raw_rows": int(len(raw)), "duplicated_texts": int(len(n_labels)),
            "duplicated_texts_with_conflicting_labels": int((n_labels > 1).sum()),
            "conflict_rate": round(float((n_labels > 1).mean()), 4),
            "examples": {t: sorted(dup.loc[dup["text"] == t, "label"].unique().tolist())
                         for t in n_labels[n_labels > 1].index[:8]}}


def tokenizer_fragmentation(meta):
    """Average number of MiniLM sub-word tokens per whitespace word, by language.
    More pieces per word = the pre-trained vocabulary fits the language worse."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    texts = pd.read_csv(MASTER_CSV, usecols=["text"]).loc[meta["row_id"], "text"].astype(str)
    out = {}
    for l in ["english", "roman_urdu"]:
        part = texts[(meta["language"] == l).values]
        n_pieces = sum(len(tok.tokenize(t)) for t in part)
        n_words = sum(len(t.split()) for t in part)
        out[l] = {"words": n_words, "subword_tokens": n_pieces,
                  "tokens_per_word": round(n_pieces / n_words, 3)}
    return out


def plot_comparison(table, path):
    groups = table["group"].unique().tolist()
    models = table["model"].unique().tolist()
    x = np.arange(len(groups))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(models):
        part = table[table["model"] == m].set_index("group").loc[groups]
        err = [part["macro_f1"] - part["f1_ci_low"], part["f1_ci_high"] - part["macro_f1"]]
        bars = ax.bar(x - 0.4 + w * (i + 0.5), part["macro_f1"], w, yerr=err, capsize=2,
                      label=m)
        ax.bar_label(bars, fmt="%.3f", fontsize=6, padding=6)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Test macro-F1 (95% bootstrap CI)")
    ax.set_title("Final comparison on the test split")
    ax.legend(fontsize=7, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_slices(table, path):
    piv = table.pivot(index="slice", columns="model", values="error_rate")
    piv = piv.loc[table["slice"].unique()]
    n = table.drop_duplicates("slice").set_index("slice")["n"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(piv))
    for i, m in enumerate([BASE, OURS]):
        bars = ax.barh(y + (i - 0.5) * 0.4, piv[m], 0.4, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_yticks(y, [f"{s} (n={n[s]})" for s in piv.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Error rate on the test split")
    ax.set_title("Where do the models fail?")
    ax.set_xlim(0, 0.7)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ensure_dirs()
    probs, y, meta = all_test_probs()
    lang = meta["language"].values
    groups = {"All test": np.ones(len(y), bool), "English": lang == "english",
              "Roman Urdu": lang == "roman_urdu"}
    rng = np.random.default_rng(SEED)
    boot_idx = {g: [rng.integers(0, m.sum(), m.sum()) for _ in range(N_BOOT)]
                for g, m in groups.items()}

    table = comparison_table(probs, y, groups, boot_idx)
    table.to_csv(os.path.join(RESULTS, "final_comparison.csv"), index=False)
    print(table.to_string(index=False))

    pairs = [(OURS, BASE), (OURS, "Majority class")] + [(m, OURS) for m in MITIGATED]
    sig = significance(probs, y, groups, boot_idx, pairs)
    sig.to_csv(os.path.join(RESULTS, "significance.csv"), index=False)
    print(sig.to_string(index=False))

    slices = failure_slices(probs, y, meta)
    slices.to_csv(os.path.join(RESULTS, "failure_slices.csv"), index=False)
    print(slices.to_string(index=False))

    stats = {"label_noise_roman_urdu": label_noise(),
             "tokenizer_fragmentation_test": tokenizer_fragmentation(meta)}
    with open(os.path.join(RESULTS, "failure_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    plot_comparison(table, os.path.join(FIG, "final_comparison.png"))
    plot_slices(slices, os.path.join(FIG, "failure_slices.png"))
    print("Saved results/final_comparison.csv, significance.csv, failure_slices.csv, "
          "failure_stats.json and the figures")


if __name__ == "__main__":
    main()
