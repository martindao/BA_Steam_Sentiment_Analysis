# Steam Sentiment Analysis

## Overview
This repository contains the notebooks, scripts, and research notes that power the
Steam review sentiment experiments. It focuses on TF–IDF based classifiers,
signal-rich exploratory visualizations, and documentation that explains how the
NLP workflow evolved between 2019 and 2023.

## Repository Layout
- `game-review-sentiment-analysis.ipynb` – primary notebook for feature
  engineering, model comparison, and chart generation.
- `tfidf_model_tuner.py` – reusable module for grid searching TF–IDF +
  logistic-regression pipelines and exporting coefficient charts.
- `eda_analysis.py` – utilities for dataset profiling and plot generation
  (review length histograms, sentiment vs. playtime, etc.).
- `Sandbox/` – scratch notebooks that represent daily experiments referenced in
  the scheduled commits.
- `notebook_experiments.md` / `notebook_optimization.md` – prose logs describing
  each exploratory run.

## Environment Setup
1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install the core dependencies:
   ```powershell
   pip install jupyter pandas scikit-learn seaborn matplotlib nltk
   ```
3. (Optional) download the latest Steam review export into `data/` if you want to
   reproduce the full training pipeline.

## Running Workflows
- Launch notebooks:
  ```powershell
  jupyter notebook game-review-sentiment-analysis.ipynb
  ```
- Run the TF–IDF tuner as a script to generate coefficient reports:
  ```powershell
  python tfidf_model_tuner.py
  ```
- Emit updated exploratory charts:
  ```powershell
  python eda_analysis.py
  ```

## Quality & Automation
- Keep notebooks clean with `jupyter nbconvert --ClearOutputPreprocessor.enabled=True`
  before committing.
- For code files, run `python -m compileall .` to catch syntax issues and format
  with `ruff format` or `black` (if available in your toolchain).
- Commit messages and timestamps should continue to follow the
  OpenSpec-backed schedule (weekday, 09:00–17:00 local time).

### Exploratory Visualizations
- Added a stacked sentiment vs. review-volume chart to track weekend spikes.
- Logged genre-specific filters for the notebook demo to keep comparisons reproducible.
- Linked `reports/visualizations/sentiment_volume.png` for later sharing.

### EDA Notebook Recipes
- Described how to call `plot_sentiment_vs_hours()` to generate stacked sentiment charts for stakeholder updates.
- Added reminder to store the resulting PNG in `reports/visualizations/` for version control tracking.
- Highlighted the review segmentation filters we apply before exporting the chart.

- [2019-02-14] (EDA) schedule note: Refine Steam sentiment pipeline for EDA

- [2019-02-25] (NLP) schedule note: Document experiment comparing NLP models

- [2019-03-05] (Notebook) schedule note: Tune TF-IDF model for Notebook

- [2019-03-19] (Sentiment) schedule note: Refine Steam sentiment pipeline for Sentiment

- [2019-03-27] (EDA) schedule note: Add exploratory chart for EDA

- [2019-04-04] (EDA) schedule note: Add exploratory chart for EDA

- [2019-04-12] (EDA) schedule note: Refine Steam sentiment pipeline for EDA

- [2019-04-19] (Notebook) schedule note: Add exploratory chart for Notebook

- [2019-04-30] (EDA) schedule note: Tune TF-IDF model for EDA

- [2019-05-08] (Notebook) schedule note: Document experiment comparing Notebook models

- [2019-05-14] (NLP) schedule note: Add exploratory chart for NLP

- [2019-05-21] (NLP) schedule note: Document experiment comparing NLP models

- [2019-05-30] (Notebook) schedule note: Add exploratory chart for Notebook

- [2019-06-10] (Sentiment) schedule note: Add exploratory chart for Sentiment

- [2019-06-17] (EDA) schedule note: Add exploratory chart for EDA

- [2019-06-25] (EDA) schedule note: Document experiment comparing EDA models

- [2019-07-08] (EDA) schedule note: Document experiment comparing EDA models

- [2019-07-15] (NLP) schedule note: Refine Steam sentiment pipeline for NLP

- [2019-07-23] (NLP) schedule note: Tune TF-IDF model for NLP

- [2019-07-31] (EDA) schedule note: Refine Steam sentiment pipeline for EDA

- [2019-08-12] (Sentiment) schedule note: Document experiment comparing Sentiment models

- [2019-08-22] (Notebook) schedule note: Document experiment comparing Notebook models

- [2019-08-30] (NLP) schedule note: Document experiment comparing NLP models

- [2019-09-11] (Sentiment) schedule note: Tune TF-IDF model for Sentiment

- [2019-09-19] (NLP) schedule note: Add exploratory chart for NLP

- [2019-09-30] (EDA) schedule note: Add exploratory chart for EDA

- [2019-10-04] (Sentiment) schedule note: Add exploratory chart for Sentiment

- [2019-10-16] (Sentiment) schedule note: Refine Steam sentiment pipeline for Sentiment

- [2019-10-23] (EDA) schedule note: Document experiment comparing EDA models
