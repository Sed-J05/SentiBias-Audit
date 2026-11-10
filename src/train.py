"""
Phase 4 - Training
Trains the SentimentMLP head on the cached MiniLM embeddings with mini-batch
AdamW, class-weighted cross-entropy and early stopping on validation macro-F1
(patience 4, best weights restored). The final hyperparameters are read from
results/tuning_results.json (written by tune.py).

Outputs:
  models/sentiment_mlp.pt          trained weights + hyperparameters
  results/training_history.json    per-epoch loss / accuracy / macro-F1
  report/figures/training_curves.png
"""
import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import EMB_NPY, FIG, LABELS, MODELS, RESULTS, SEED, SPLITS_CSV, ensure_dirs
from model import SentimentMLP, build_model

DEFAULT_PARAMS = {"lr": 1e-3, "batch_size": 128, "dropout": 0.3, "hidden": 256,
                  "weight_decay": 1e-4}


def load_split_arrays():
    """Return {split: (X, y, meta DataFrame)} for train/val/test/bias."""
    emb = np.load(EMB_NPY)
    splits = pd.read_csv(SPLITS_CSV)
    label_id = {lbl: i for i, lbl in enumerate(LABELS)}
    data = {}
    for name, part in splits.groupby("split"):
        part = part.reset_index(drop=True)
        X = torch.tensor(emb[part["row_id"].values])
        y = torch.tensor(part["label"].map(label_id).values, dtype=torch.long)
        data[name] = (X, y, part)
    return data


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.softmax(model(X), dim=1).numpy()


def run_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_split(model, X, y, loss_fn):
    model.eval()
    logits = model(X)
    pred = logits.argmax(1)
    return (loss_fn(logits, y).item(), (pred == y).float().mean().item(),
            f1_score(y.numpy(), pred.numpy(), average="macro"))


def train_model(params, data, max_epochs=40, patience=4, verbose=True):
    """Train one model; returns (best model, history list, best val macro-F1)."""
    torch.manual_seed(SEED)
    X_tr, y_tr, _ = data["train"]
    X_va, y_va, _ = data["val"]
    model, loss_fn, optimizer = build_model(params, y_tr.numpy())
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=params["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(SEED))

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loader, loss_fn, optimizer)
        va_loss, va_acc, va_f1 = evaluate_split(model, X_va, y_va, loss_fn)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1})
        if verbose:
            print(f"epoch {epoch:2d}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} macro-F1 {va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1, best_state, bad_epochs = va_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
                break
    model.load_state_dict(best_state)
    return model, history, best_f1


def plot_curves(history, path):
    h = pd.DataFrame(history)
    best = h.loc[h["val_macro_f1"].idxmax(), "epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(h["epoch"], h["train_loss"], marker="o", label="train")
    axes[0].plot(h["epoch"], h["val_loss"], marker="o", label="validation")
    axes[0].set_title("Weighted cross-entropy loss")
    axes[1].plot(h["epoch"], h["train_acc"], marker="o", label="train accuracy")
    axes[1].plot(h["epoch"], h["val_acc"], marker="o", label="validation accuracy")
    axes[1].plot(h["epoch"], h["val_macro_f1"], marker="s", linestyle="--",
                 label="validation macro-F1")
    axes[1].set_title("Accuracy / macro-F1")
    for ax in axes:
        ax.axvline(best, color="grey", linestyle=":", label="best epoch")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ensure_dirs()
    tuning_path = os.path.join(RESULTS, "tuning_results.json")
    params = DEFAULT_PARAMS
    if os.path.exists(tuning_path):
        with open(tuning_path, encoding="utf-8") as f:
            params = json.load(f)["best_params"]
    else:
        print("tuning_results.json not found - using default hyperparameters")
    print("Hyperparameters:", params)

    data = load_split_arrays()
    model, history, best_f1 = train_model(params, data)

    torch.save({"state_dict": model.state_dict(), "params": params},
               os.path.join(MODELS, "sentiment_mlp.pt"))
    with open(os.path.join(RESULTS, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump({"params": params, "best_val_macro_f1": best_f1, "history": history},
                  f, indent=2)
    plot_curves(history, os.path.join(FIG, "training_curves.png"))
    print(f"Saved model to models/sentiment_mlp.pt (best val macro-F1 {best_f1:.4f})")


def load_trained_model():
    ckpt = torch.load(os.path.join(MODELS, "sentiment_mlp.pt"), weights_only=False)
    p = ckpt["params"]
    model = SentimentMLP(hidden=p["hidden"], dropout=p["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    main()
