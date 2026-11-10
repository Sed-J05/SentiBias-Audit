# SentiBias Audit

A research project that tests multilingual sentiment models for bias across dialects,
gender-coded names, and code-switched text (Roman Urdu and English), then applies
mitigation methods to reduce the performance gaps.

* **Deliverable 1 – Problem Definition & Data Foundations** (Phase 1 and Phase 2): `data_scripts/`
* **Deliverable 2 – Model Architecture & Empirical Evaluation** (Phase 3 and Phase 4): `src/`

## Datasets

| Dataset | Bias dimension | Size (raw) | Source |
|---|---|---|---|
| Roman Urdu Sentiment | Roman Urdu / code-switched text | 20,229 | [UCI ML Repository #458](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set) |
| TweetEval Sentiment | English baseline (tweets) | 59,899 | [Cardiff NLP, GitHub](https://github.com/cardiffnlp/tweeteval) |
| Equity Evaluation Corpus (EEC) | Gender-coded names | 8,640 | [Kiritchenko & Mohammad, 2018](https://saifmohammad.com/WebPages/Biases-SA.html) |

Raw data is **not** committed (see `.gitignore`); it is re-downloaded by the script below.
The EEC terms of use forbid redistribution, so it must always be fetched from the original page.

## Repository structure

```
.
├── data_scripts/
│   ├── download_data.py   # Phase 2 – collection (writes data/raw/ + download_log.txt)
│   ├── clean_data.py      # Phase 2 – missing values, duplicates, outliers
│   └── feature_eda.py     # Phase 2 – normalisation, encoding, scaling, EDA charts
├── src/
│   ├── config.py          # shared paths, labels, random seed
│   ├── split_data.py      # Phase 4 – stratified 70/15/15 split, EEC held out for bias check
│   ├── embed.py           # Phase 3 – frozen multilingual MiniLM sentence embeddings
│   ├── model.py           # Phase 3 – SentimentMLP (PyTorch), weighted loss, AdamW
│   ├── train.py           # Phase 4 – training loop, early stopping, loss/accuracy curves
│   ├── tune.py            # Phase 3 – hyperparameter grid search
│   ├── baseline.py        # Phase 3 – TF-IDF + Logistic Regression baseline (scikit-learn)
│   ├── evaluate.py        # Phase 4 – P/R/F1/AUC-ROC, subgroups, error analysis
│   └── bias_check.py      # Phase 4 – EEC gender/race bias check
├── results/               # metric tables (JSON/CSV) written by src/
├── models/                # trained weights (git-ignored, see link below)
├── data/                  # raw + processed data (git-ignored)
├── report/                # LaTeX reports (report = D1, deliverable2 = D2), PDFs, figures
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
# Phase 2 (data)
python data_scripts/download_data.py
python data_scripts/clean_data.py
python data_scripts/feature_eda.py
# Phase 3/4 (model + evaluation)
python src/split_data.py     # train/val/test split
python src/embed.py          # ~25 min on CPU, run once
python src/tune.py           # grid search (24 configurations, ~25 min on CPU)
python src/train.py          # final model with the best hyperparameters
python src/baseline.py       # TF-IDF + Logistic Regression baseline
python src/evaluate.py       # test metrics + figures
python src/bias_check.py     # EEC bias check
```

To rebuild the reports: `cd report && pdflatex report.tex` (Deliverable 1) and
`pdflatex deliverable2.tex` (Deliverable 2), or use `tectonic` instead of `pdflatex`.

## Model weights

Weights are not stored in Git. Download them from:
**[Google Drive – SentiBias model weights](https://drive.google.com/drive/folders/1Ynhwlni6P-MIlXFW4p-vFwvR8txQxNHE?usp=drive_link)**

Put both files in `models/`:

* `sentiment_mlp.pt` – trained MLP classification head (PyTorch)
* `baseline_tfidf_lr.joblib` – TF-IDF + Logistic Regression baseline

The frozen encoder is downloaded automatically from Hugging Face:
[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

## Results (test split)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Macro AUC-ROC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.621 | 0.612 | 0.633 | 0.616 | 0.806 |
| MiniLM + MLP (ours) | **0.643** | **0.633** | **0.652** | **0.638** | **0.828** |

Final MLP hyperparameters (from a 24-configuration grid search): learning rate 1e-4, batch size 64,
dropout 0.2, hidden size 512, AdamW with weight decay 1e-4, early stopping at epoch 11.

**Bias Check (summary):** macro-F1 is 0.671 on English vs. 0.535 on Roman Urdu for the MLP, a language gap
of 0.136, more than twice the baseline's 0.056. On the EEC, both models show a statistically significant gender
score gap (paired t-test, p < 1e-10), in opposite directions; the MLP's race gap is not significant (p = 0.52).
Details are in `report/deliverable2.pdf` and `results/`.

## Processed data schema

Every cleaned file shares the columns
`text, label (negative/neutral/positive), language, source, split, gender,
code_switched, n_tokens, n_chars, english_word_ratio`.
`sentiment_master.csv` concatenates all three datasets (87,696 rows after cleaning).

## Cleaning summary

| Dataset | Raw | After missing-value removal | After de-duplication | After outlier removal |
|---|---|---|---|---|
| Roman Urdu | 20,229 | 20,116 | 19,582 | 19,187 |
| TweetEval | 59,899 | 59,899 | 59,869 | 59,869 |
| EEC | 8,640 | 8,640 | 8,640 | 8,640 |

## References

- Roman Urdu Data Set, UCI Machine Learning Repository (2018).
- Barbieri et al. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*. Findings of EMNLP.
- Kiritchenko & Mohammad (2018). *Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems*. \*SEM.
- Reimers & Gurevych (2019/2020). *Sentence-BERT* and *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*.
