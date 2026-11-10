"""
Shared paths and constants for the Phase 3/4 modelling code.
Every script in src/ imports from here so that paths and the random seed
are defined in exactly one place.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")

MASTER_CSV = os.path.join(PROC, "sentiment_master_normalized.csv")  # from feature_eda.py
SPLITS_CSV = os.path.join(PROC, "splits.csv")                       # from split_data.py
EMB_NPY = os.path.join(PROC, "embeddings_minilm.npy")               # from embed.py

# Frozen multilingual transformer used as the sentence encoder
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_DIM = 384

LABELS = ["negative", "neutral", "positive"]   # same order as LabelEncoder in Phase 2
SEED = 42


def ensure_dirs():
    for d in (MODELS, RESULTS, FIG):
        os.makedirs(d, exist_ok=True)
