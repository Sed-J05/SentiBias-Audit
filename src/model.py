"""
Phase 3 - Model architecture (PyTorch)

Full pipeline:
  text -> frozen multilingual MiniLM encoder (12 transformer layers, mean pooling)
       -> 384-d sentence embedding
       -> SentimentMLP (trainable classification head, defined below)
       -> softmax over {negative, neutral, positive}

SentimentMLP layers:
  Linear(384 -> h) -> ReLU -> Dropout(p)
  Linear(h -> h/2) -> ReLU -> Dropout(p)
  Linear(h/2 -> 3)                      (logits; softmax is applied inside the loss)

Loss: class-weighted cross-entropy (weights = inverse class frequency in train)
Optimizer: AdamW (Adam with decoupled weight decay)

Phase 5 (ablation study) adds two options with defaults that give exactly the
model above: n_layers (number of hidden layers: 2, 1 or 0 = linear head) and
class_weighted (False = plain cross-entropy).
"""
import numpy as np
import torch
from torch import nn


class SentimentMLP(nn.Module):
    def __init__(self, in_dim=384, hidden=256, dropout=0.3, n_classes=3, n_layers=2):
        super().__init__()
        sizes = [in_dim, hidden, hidden // 2][:n_layers + 1]   # hidden sizes: h, h/2
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(sizes[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def class_weights(y, n_classes=3):
    """Inverse-frequency weights, normalised so they average to 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def build_model(params, y_train):
    """Create the model, the weighted loss and the optimizer from a hyperparameter dict."""
    model = SentimentMLP(in_dim=params.get("in_dim", 384), hidden=params["hidden"],
                         dropout=params["dropout"], n_layers=params.get("n_layers", 2))
    weights = class_weights(y_train) if params.get("class_weighted", True) else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                  weight_decay=params.get("weight_decay", 1e-4))
    return model, loss_fn, optimizer
