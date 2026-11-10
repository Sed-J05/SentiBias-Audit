"""
Phase 3 - Sentence embeddings
Encodes every row of the master dataset (Roman Urdu, TweetEval and EEC) with
a frozen pre-trained multilingual transformer (paraphrase-multilingual-MiniLM-L12-v2,
12 layers, 384-d output, mean pooling). The embeddings are cached to
data/processed/embeddings_minilm.npy, one row per row of the master CSV, so the
encoder only has to run once on CPU.
"""
import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import EMB_NPY, ENCODER_NAME, MASTER_CSV


def main():
    df = pd.read_csv(MASTER_CSV)
    texts = df["text"].astype(str).tolist()        # raw text: the tokenizer handles casing/punctuation
    print(f"Encoding {len(texts)} texts with {ENCODER_NAME} ...")

    encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
    start = time.time()
    emb = encoder.encode(texts, batch_size=128, show_progress_bar=True,
                         convert_to_numpy=True)
    np.save(EMB_NPY, emb.astype(np.float32))
    print(f"Saved {emb.shape} to {os.path.relpath(EMB_NPY)} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
