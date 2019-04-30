# Steam Sentiment Analysis

This project analyzes sentiment in Steam game reviews using various NLP techniques.

## Overview

The project explores different approaches to sentiment classification and analysis of game review text.

### Sentiment Analysis Models

Experiment comparing different sentiment analysis approaches:
- Traditional ML models (TF-IDF + classifiers)
- Pre-trained transformer models
- Custom neural networks

## Files

- `game-review-sentiment-analysis.ipynb` - Main analysis notebook
- `Sandbox/` - Additional experimental files

## Getting Started

To run the analysis:

1. Install required dependencies
2. Run the Jupyter notebook
3. Explore the sentiment analysis results

## License

This project is for educational purposes.
"" 

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
