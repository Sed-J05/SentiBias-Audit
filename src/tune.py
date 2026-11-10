"""
Phase 3 - Hyperparameter tuning
Full grid search over learning rate, batch size, dropout and hidden size.
Every configuration is trained on the train split with early stopping and
scored on the validation split (macro-F1). The test split is never touched here.

Outputs:
  results/tuning_results.csv   one row per configuration
  results/tuning_results.json  best configuration (read by train.py)
"""
import itertools
import json
import os
import time

import pandas as pd

from config import RESULTS, ensure_dirs
from train import load_split_arrays, train_model

GRID = {
    "lr": [1e-2, 1e-3, 1e-4],
    "batch_size": [64, 256],
    "dropout": [0.2, 0.5],
    "hidden": [256, 512],
}
WEIGHT_DECAY = 1e-4


def main():
    ensure_dirs()
    data = load_split_arrays()
    rows = []
    combos = list(itertools.product(*GRID.values()))
    for i, values in enumerate(combos, 1):
        params = dict(zip(GRID.keys(), values), weight_decay=WEIGHT_DECAY)
        start = time.time()
        _, history, best_f1 = train_model(params, data, max_epochs=25, verbose=False)
        best = max(history, key=lambda h: h["val_macro_f1"])
        rows.append({**params, "best_epoch": best["epoch"], "epochs_run": len(history),
                     "val_acc": round(best["val_acc"], 4), "val_macro_f1": round(best_f1, 4),
                     "seconds": round(time.time() - start, 1)})
        print(f"[{i:2d}/{len(combos)}] {params} -> val macro-F1 {best_f1:.4f} "
              f"(epoch {best['epoch']})")

    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    table.to_csv(os.path.join(RESULTS, "tuning_results.csv"), index=False)
    best = table.iloc[0]
    best_params = {"lr": float(best["lr"]), "batch_size": int(best["batch_size"]),
                   "dropout": float(best["dropout"]), "hidden": int(best["hidden"]),
                   "weight_decay": WEIGHT_DECAY}
    with open(os.path.join(RESULTS, "tuning_results.json"), "w", encoding="utf-8") as f:
        json.dump({"grid": GRID, "best_params": best_params,
                   "best_val_macro_f1": float(best["val_macro_f1"])}, f, indent=2)
    print("\nTop 5 configurations:\n", table.head().to_string(index=False))
    print("Best:", best_params)


if __name__ == "__main__":
    main()
