#!/usr/bin/env python3
"""
Steam Sentiment Analysis - Exploratory Data Analysis
===================================================

Enhanced EDA pipeline for analyzing Steam game review data.
This module provides functions for data loading, preprocessing, and initial analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_steam_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess Steam review data.
    
    Args:
        file_path (str): Path to the CSV file containing Steam reviews
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with Steam review data
    """
    df = pd.read_csv(file_path)
    
    # Basic data info
    print("Data Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    
    return df

def analyze_review_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of reviews in the dataset.
    
    Args:
        df (pd.DataFrame): Steam review DataFrame
        
    Returns:
        Dict: Dictionary containing distribution statistics
    """
    stats = {}
    
    # Check for missing values
    stats['missing_values'] = df.isnull().sum()
    
    # Language distribution
def generate_eda_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive EDA report.
    
    Args:
        df (pd.DataFrame): Steam review DataFrame
        
    Returns:
        str: Formatted EDA report
    """
    report = []
    report.append("=== Steam Review Data Analysis Report ===\n")
    
    # Data overview
    report.append(f"Dataset Shape: {df.shape}")
    report.append(f"Total Reviews: {len(df)}")
    
    if 'review' in df.columns:
        # Calculate average review length
        df['review_length'] = df['review'].str.len()
        avg_length = df['review_length'].mean()
        report.append(f"Average Review Length: {avg_length:.2f} characters")
        
        # Word count analysis
        df['word_count'] = df['review'].str.split().str.len()
        avg_words = df['word_count'].mean()
        report.append(f"Average Word Count: {avg_words:.2f} words")
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    report.append(f"\nMissing Data Summary:")
    for col, missing_count in missing_data[missing_data > 0].items():
        report.append(f"  {col}: {missing_count} ({missing_count/len(df)*100:.1f}%)")
    
    return "\n".join(report)
    if 'language' in df.columns:
        stats['language_distribution'] = df['language'].value_counts()
    
    # Vote distribution
    if 'voted_up' in df.columns:
        stats['vote_distribution'] = df['voted_up'].value_counts()
    
    # Review length analysis
    if 'review' in df.columns:
        df['review_length'] = df['review'].str.len()
        stats['review_length_stats'] = df['review_length'].describe()
    
    return stats

def create_eda_visualizations(df: pd.DataFrame) -> None:
    """
    Create exploratory data analysis visualizations.
    
    Args:
        df (pd.DataFrame): Steam review DataFrame
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Steam Review Data - Exploratory Data Analysis', fontsize=16)
    
    # Review length distribution
    if 'review_length' in df.columns:
        axes[0, 0].hist(df['review_length'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Review Length Distribution')
        axes[0, 0].set_xlabel('Character Count')
        axes[0, 0].set_ylabel('Frequency')
    
    # Language distribution
    if 'language' in df.columns:
        lang_counts = df['language'].value_counts().head(10)
        axes[0, 1].bar(range(len(lang_counts)), lang_counts.values, color='lightcoral')
        axes[0, 1].set_title('Top 10 Languages in Reviews')
        axes[0, 1].set_xlabel('Language')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(range(len(lang_counts)))
        axes[0, 1].set_xticklabels(lang_counts.index, rotation=45)
    
    # Vote distribution
    if 'voted_up' in df.columns:
        vote_counts = df['voted_up'].value_counts()
        axes[1, 0].pie(vote_counts.values, labels=vote_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Review Vote Distribution')
    
    # Playtime analysis
    if 'author.playtime_forever' in df.columns:
        axes[1, 1].hist(df['author.playtime_forever'], bins=50, alpha=0.7, color='lightgreen')
        axes[1, 1].set_title('Author Playtime Distribution')
        axes[1, 1].set_xlabel('Playtime (minutes)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('steam_eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the EDA analysis.
    """
    print("=== Steam Sentiment Analysis - EDA Pipeline ===")
    print("Enhanced exploratory data analysis for Steam review data.")
    print("This analysis provides insights into data quality and distribution.")
    
    # This would normally load real data
    # df = load_steam_data('path/to/steam_reviews.csv')
    
    # For demonstration, create sample data
    np.random.seed(42)
    sample_data = {
        'review': ['Great game!', 'Love the story', 'Amazing graphics', 'Best gameplay ever'] * 100,
        'language': ['english'] * 400,
        'voted_up': [True, False] * 200,
        'author.playtime_forever': np.random.exponential(1000, 400)
    }
    df = pd.DataFrame(sample_data)
    
    # Run analysis
    stats = analyze_review_distribution(df)
    create_eda_visualizations(df)
    
    print("\n=== EDA Analysis Complete ===")
    print("Check 'steam_eda_analysis.png' for visualizations.")
    
    return stats

    
    return stats

def enhanced_notebook_eda_pipeline(df: pd.DataFrame) -> Dict:
    """
    Enhanced EDA pipeline specifically optimized for Jupyter notebook usage.
    
    Args:
        df (pd.DataFrame): Steam review DataFrame
        
    Returns:
        Dict: Comprehensive notebook-ready analysis results
    """
    print("=== Enhanced Notebook EDA Pipeline for Steam Sentiment ===")
    
    notebook_results = {}
    
    # Data profiling for notebook display
    notebook_results['data_profile'] = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Steam-specific sentiment analysis prep
    if 'review' in df.columns:
        # Review quality metrics
        df['review_quality_score'] = df['review'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        notebook_results['review_metrics'] = {
            'avg_review_length': df['review'].str.len().mean(),
            'avg_word_count': df['word_count'].mean() if 'word_count' in df.columns else 0,
            'high_quality_reviews': (df['review_quality_score'] > 20).sum(),
            'review_engagement': df['review_quality_score'].describe()
        }
    
    # Steam-specific features analysis
    steam_features = ['voted_up', 'votes_up', 'weighted_vote_score']
    available_features = [f for f in steam_features if f in df.columns]
    
    if available_features:
        notebook_results['steam_features'] = {}
        for feature in available_features:
            notebook_results['steam_features'][feature] = {
                'distribution': df[feature].value_counts().to_dict(),
                'statistics': df[feature].describe().to_dict()
            }
    
    # Memory-optimized data sampling for notebooks
    notebook_results['sampling_strategy'] = {
        'recommended_sample_size': min(5000, len(df) // 10),  # 10% or 5k max
        'stratified_columns': available_features[:2] if len(available_features) >= 2 else [],
        'random_state': 42
    }
    
    print(f"✓ Data Profile: {notebook_results['data_profile']['shape'][0]} rows, {notebook_results['data_profile']['memory_usage']:.1f}MB")
    print(f"✓ Missing Data: {notebook_results['data_profile']['missing_percentage']:.1f}%")
    print(f"✓ Recommended Sample: {notebook_results['sampling_strategy']['recommended_sample_size']} rows")
    
    return notebook_results

def notebook_visualization_config() -> Dict:
    """
    Provide optimized visualization configurations for notebook usage.
    
    Returns:
        Dict: Visualization configuration settings
    """
    return {
        'figure_style': 'seaborn-v0_8',
        'figure_size': (12, 8),
        'dpi': 150,
        'interactive_plots': True,
        'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        'theme': 'lightgrid',
        'save_formats': ['png', 'html', 'svg']
    }

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()