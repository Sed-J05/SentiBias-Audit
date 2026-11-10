"""
Phase 4 - Validation strategy
Creates one fixed, stratified 70/15/15 train/validation/test split of the two
labelled corpora (Roman Urdu + TweetEval). Stratification is on language x label,
so every split has the same language mix and the same class balance.
The EEC corpus is NOT used for training: all of its rows are assigned to the
held-out "bias" split and are only used for the Bias Check.

Output: data/processed/splits.csv with one row per master row:
  row_id (index into the master CSV / embedding matrix), split, label, language,
  code_switched, n_tokens
"""
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import MASTER_CSV, PROC, SEED, SPLITS_CSV


def main():
    df = pd.read_csv(MASTER_CSV)
    df["row_id"] = df.index
    is_eec = df["source"].str.startswith("EEC")

    labelled = df[~is_eec]
    strata = labelled["language"] + "_" + labelled["label"]

    # 70% train, then split the remaining 30% in half -> 15% val, 15% test
    train_ids, rest_ids = train_test_split(labelled["row_id"], test_size=0.30,
                                           stratify=strata, random_state=SEED)
    val_ids, test_ids = train_test_split(rest_ids, test_size=0.50,
                                         stratify=strata.loc[rest_ids], random_state=SEED)

    df["model_split"] = "bias"
    df.loc[train_ids, "model_split"] = "train"
    df.loc[val_ids, "model_split"] = "val"
    df.loc[test_ids, "model_split"] = "test"

    out = df[["row_id", "model_split", "label", "language", "code_switched", "n_tokens"]]
    out = out.rename(columns={"model_split": "split"})
    out.to_csv(SPLITS_CSV, index=False)

    summary = pd.crosstab([out["split"], out["language"]], out["label"])
    print(summary)
    counts = {s: int(n) for s, n in out["split"].value_counts().items()}
    with open(os.path.join(PROC, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"rows_per_split": counts,
                   "by_split_language_label": {f"{a}|{b}": row.to_dict()
                                               for (a, b), row in summary.iterrows()}},
                  f, indent=2)
    print(counts)


if __name__ == "__main__":
    main()
