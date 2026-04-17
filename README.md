# Steam Sentiment Analysis

## Overview

This repository provides data quality validation and exploratory analysis for Steam review datasets. It focuses on reproducible validation checks, dataset profiling, and documentation that supports the broader data quality story. The NLP experiments here serve as a downstream consumer of validated data.

## Quick Start

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install jupyter pandas scikit-learn seaborn matplotlib nltk

# 3. Validate your data
python validate_data.py

# 4. Launch the primary notebook
jupyter notebook game-review-sentiment-analysis.ipynb
```

## Repository Layout

| File | Purpose |
|------|---------|
| `game-review-sentiment-analysis.ipynb` | Primary notebook for feature engineering, model comparison, and chart generation |
| `tfidf_model_tuner.py` | Grid search module for TF-IDF + logistic regression pipelines with coefficient chart exports |
| `eda_analysis.py` | Dataset profiling utilities (review length histograms, sentiment vs. playtime, etc.) |
| `Sandbox/` | Scratch notebooks representing daily experiments referenced in the historical log |
| `notebook_experiments.md` | Prose log describing each exploratory run |
| `notebook_optimization.md` | Prose log for optimization experiments |

## Dataset Quality Controls

Before any modeling, we validate the input dataset against expected schemas and quality thresholds.

**Required columns:** `review_text`, `sentiment_label`, `playtime_hours`, `review_date`

**Validation checks:**

| Check | Description |
|-------|-------------|
| Schema validation | Ensures all required columns exist |
| Null-rate logging | Flags columns exceeding 5% missing values |
| Duplicate detection | Identifies duplicates via `review_text` + `user_id` composite key |
| Playtime outliers | Flags values exceeding 3 standard deviations from the mean |

**Artifact location:** Validation reports are stored in `reports/data-quality/`

## Validation Before Modeling

Run the validation script before training any models:

```powershell
python validate_data.py
```

This generates:
- `reports/data-quality/schema-check-report.md` - Column presence and type validation
- `reports/data-quality/null-rate-summary.csv` - Missing value percentages per column
- `reports/data-quality/dataset-validation-notes.md` - Pass/fail summary with recommendations

If validation fails, the script exits with a non-zero code and logs which checks failed.

## How This Repository Complements the Main Data Stack

This repository is a supporting asset, not a standalone data platform. It provides:

- **Validation artifacts** that prove data quality before downstream analysis
- **Exploratory analysis** that informs feature engineering decisions
- **Reproducible notebooks** that document the experimentation process

The primary data stack ownership remains in the main analytics engineering repository. This repo exists to validate and explore, not to own production data pipelines or MLOps infrastructure.

## Quality & Automation

**Notebooks:** Clear outputs before pushing changes:
```powershell
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace game-review-sentiment-analysis.ipynb
```

**Code files:** Check syntax and format:
```powershell
python -m compileall .
ruff format .  # or: black .
```

### Exploratory Visualizations

- Stacked sentiment vs. review-volume chart tracks weekend spikes
- Genre-specific filters for notebook demos keep comparisons reproducible
- Outputs saved to `reports/visualizations/sentiment_volume.png`

### EDA Notebook Recipes

Use `plot_sentiment_vs_hours()` to generate stacked sentiment charts for stakeholder updates. Store resulting PNGs in `reports/visualizations/` for version control tracking. Apply review segmentation filters before exporting charts.
