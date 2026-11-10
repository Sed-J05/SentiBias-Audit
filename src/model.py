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
"""
import numpy as np
import torch
from torch import nn


class SentimentMLP(nn.Module):
    def __init__(self, in_dim=384, hidden=256, dropout=0.3, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def class_weights(y, n_classes=3):
    """Inverse-frequency weights, normalised so they average to 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def build_model(params, y_train):
    """Create the model, the weighted loss and the optimizer from a hyperparameter dict."""
    model = SentimentMLP(hidden=params["hidden"], dropout=params["dropout"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train))
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                  weight_decay=params.get("weight_decay", 1e-4))
    return model, loss_fn, optimizer
