"""
Phase 3 - Baseline model (scikit-learn)
TF-IDF (unigrams + bigrams, 5,000 features, min_df=2, same settings as Phase 2)
+ multinomial Logistic Regression with balanced class weights.
Unlike Phase 2, the TF-IDF vocabulary is now fitted on the TRAIN split only, so no
information from validation/test/EEC leaks into the features.
The regularisation strength C is tuned on the validation split (macro-F1).

Output: models/baseline_tfidf_lr.joblib  (vectorizer + classifier)
        results/baseline_tuning.csv
"""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from config import LABELS, MASTER_CSV, MODELS, RESULTS, SEED, SPLITS_CSV, ensure_dirs

C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]


def load_text_splits():
    """Return {split: (normalised texts, integer labels)} for train/val/test/bias."""
    master = pd.read_csv(MASTER_CSV, usecols=["text_norm"])
    splits = pd.read_csv(SPLITS_CSV)
    label_id = {lbl: i for i, lbl in enumerate(LABELS)}
    out = {}
    for name, part in splits.groupby("split"):
        texts = master.loc[part["row_id"], "text_norm"].fillna("").tolist()
        out[name] = (texts, part["label"].map(label_id).values)
    return out


def make_pipeline_for(C):
    return make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def main():
    ensure_dirs()
    data = load_text_splits()
    X_tr, y_tr = data["train"]
    X_va, y_va = data["val"]

    rows, best_pipe, best_f1 = [], None, -1.0
    for C in C_GRID:
        pipe = make_pipeline_for(C).fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        f1 = f1_score(y_va, pred, average="macro")
        rows.append({"C": C, "val_acc": round(accuracy_score(y_va, pred), 4),
                     "val_macro_f1": round(f1, 4)})
        print(f"C={C:<5} val acc {rows[-1]['val_acc']:.4f}  macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_pipe, best_f1 = pipe, f1

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "baseline_tuning.csv"), index=False)
    joblib.dump(best_pipe, os.path.join(MODELS, "baseline_tfidf_lr.joblib"))
    print(f"Best C = {best_pipe[-1].C} (val macro-F1 {best_f1:.4f}); "
          "saved models/baseline_tfidf_lr.joblib")


if __name__ == "__main__":
    main()
