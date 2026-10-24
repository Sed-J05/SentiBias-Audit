# SentiBias Audit

A research project that tests multilingual sentiment models for bias across dialects,
gender-coded names, and code-switched text (Roman Urdu and English), then applies
mitigation methods to reduce the performance gaps.

**Deliverable 1 – Problem Definition & Data Foundations** (Phase 1 and Phase 2).

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
├── data/
│   ├── raw/               # downloaded files (git-ignored)
│   └── processed/         # cleaned CSVs + feature files (git-ignored)
├── report/
│   ├── report.tex         # LaTeX source of the technical report
│   ├── report.pdf         # compiled report
│   └── figures/           # EDA charts produced by feature_eda.py
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python data_scripts/download_data.py   # 1. collect raw data
python data_scripts/clean_data.py      # 2. clean -> data/processed/*.csv
python data_scripts/feature_eda.py     # 3. features + charts -> report/figures/
```

To rebuild the report: `cd report && pdflatex report.tex` (or `tectonic report.tex`).

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
