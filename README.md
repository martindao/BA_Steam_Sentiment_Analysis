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
  the historical log.
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
  before pushing changes.
- For code files, run `python -m compileall .` to catch syntax issues and format
  with `ruff format` or `black` (if available in your toolchain).

### Exploratory Visualizations
- Added a stacked sentiment vs. review-volume chart to track weekend spikes.
- Logged genre-specific filters for the notebook demo to keep comparisons reproducible.
- Linked `reports/visualizations/sentiment_volume.png` for later sharing.

### EDA Notebook Recipes
- Described how to call `plot_sentiment_vs_hours()` to generate stacked sentiment charts for stakeholder updates.
- Added reminder to store the resulting PNG in `reports/visualizations/` for version control tracking.
- Highlighted the review segmentation filters we apply before exporting the chart.
