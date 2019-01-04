#!/usr/bin/env python3
"""
Steam Sentiment Analysis - TF-IDF Model Tuning
==============================================

TF-IDF (Term Frequency-Inverse Document Frequency) model implementation
for Steam sentiment analysis with hyperparameter optimization.

This module provides TF-IDF vectorization with configurable parameters
for optimizing sentiment classification performance.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class TfidfModelTuner:
    """
    TF-IDF model tuning class for Steam sentiment analysis.
    
    This class handles the configuration, training, and optimization
    of TF-IDF vectorizers for sentiment classification tasks.
    """
    
    def __init__(self, max_features: int = 2000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TF-IDF model tuner.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF vectorizer
            ngram_range (Tuple[int, int]): N-gram range for vectorization
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.best_params = None
        self.best_score = None
        
    def create_vectorizer(self, **kwargs) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer with specified parameters.
        
        Args:
            **kwargs: Additional parameters for TfidfVectorizer
            
        Returns:
            TfidfVectorizer: Configured TF-IDF vectorizer
        """
        default_params = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'stop_words': 'english',
            'lowercase': True,
            'strip_accents': 'unicode',
            'max_df': 0.95,
            'min_df': 2
        }
        default_params.update(kwargs)
        
        return TfidfVectorizer(**default_params)
    
    def tune_model(self, X_train, y_train, param_grid: Dict = None) -> Dict[str, Any]:
        """
        Tune TF-IDF model hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            param_grid: Parameter grid for tuning
            
        Returns:
            Dict[str, Any]: Tuning results including best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'tfidf__max_features': [1000, 2000, 3000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'tfidf__min_df': [1, 2, 3],
                'tfidf__max_df': [0.8, 0.9, 0.95],
                'classifier__C': [0.1, 1.0, 10.0]
            }
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting TF-IDF model hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        
        return results
    
    def evaluate_model(self, X_train, X_test, y_train, y_test) -> Dict[str, float]:
        """
        Evaluate tuned model performance.
        
        Args:
            X_train, X_test: Training and test text data
            y_train, y_test: Training and test labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.best_params is None:
            raise ValueError("Model must be tuned before evaluation")
        
        # Create pipeline with best parameters
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.set_params(**self.best_params)
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
        }
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return results
    
    def plot_feature_importance(self, pipeline, feature_names: List[str], top_n: int = 20):
        """
        Plot top feature importances from the logistic regression model.
        
        Args:
            pipeline: Trained pipeline
            feature_names: List of feature names
            top_n: Number of top features to display
        """
        # Get feature importance (coefficients for logistic regression)
        coefficients = pipeline.named_steps['classifier'].coef_[0]
        
        # Get top features
        top_indices = np.argsort(np.abs(coefficients))[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_coefficients = coefficients[top_indices]
        
        # Plot
        plt.figure(figsize=(12, 8))
        colors = ['red' if c < 0 else 'blue' for c in top_coefficients]
        plt.barh(range(len(top_features)), top_coefficients, color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} TF-IDF Features for Sentiment Classification')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return top_features, top_coefficients

def demo_tfidf_tuning():
    """
    Demonstration of TF-IDF model tuning for Steam sentiment analysis.
    """
    print("=== Steam Sentiment Analysis - TF-IDF Model Tuning Demo ===")
    
    # Sample data for demonstration
    sample_reviews = [
        "Amazing game with great gameplay and story",
        "Terrible graphics and boring gameplay", 
        "Love the music and art style",
        "Awful combat mechanics",
        "Beautiful visuals and engaging narrative",
        "Poor optimization and frame drops",
        "Excellent storytelling and character development",
        "Lacks depth and repetitive gameplay"
    ] * 50  # Expand for demo
    
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0] * 50
    
    # Create tuner
    tuner = TfidfModelTuner(max_features=1000)
    
    # Simplified parameter grid for demo
    param_grid = {
        'tfidf__max_features': [500, 1000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [1.0, 10.0]
    }
    
    # Simulate train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        sample_reviews, sample_labels, test_size=0.2, random_state=42
    )
    
    # Tune model
    results = tuner.tune_model(X_train, y_train, param_grid)
    
    # Evaluate model
    eval_results = tuner.evaluate_model(X_train, X_test, y_train, y_test)
    
    print("\n=== TF-IDF Model Tuning Complete ===")
    print(f"Best cross-validation score: {results['best_score']:.4f}")
    print(f"Test accuracy: {eval_results['test_accuracy']:.4f}")
    
    return tuner, results, eval_results

if __name__ == "__main__":
    demo_tfidf_tuning()