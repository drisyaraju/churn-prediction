"""
Configuration file for Customer Churn Prediction Project
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Kaggle dataset configuration
KAGGLE_DATASET = "blastchar/telco-customer-churn"
DATASET_FILENAME = "Telco-Customer-Churn.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

# Alternative direct URL (GitHub raw)
GITHUB_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Data preprocessing configuration
TARGET_COLUMN = 'Churn'
DROP_COLUMNS = ['customerID']
NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Train-test split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model hyperparameters - UPDATED TO PREVENT OVERFITTING
MODEL_PARAMS = {
    'logistic_regression': {
        'random_state': RANDOM_STATE,
        'max_iter': 1000,
        'C': 0.1,  # Reduced regularization strength
        'class_weight': 'balanced',
        'penalty': 'l2'
    },
    'decision_tree': {
        'random_state': RANDOM_STATE,
        'max_depth': 5,  # Limit tree depth
        'min_samples_split': 20,  # Minimum samples to split
        'min_samples_leaf': 10,  # Minimum samples in leaf
        'max_features': 'sqrt',  # Consider subset of features
        'class_weight': 'balanced'
    },
    'random_forest': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'max_depth': 8,  # Limit depth
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'bootstrap': True,
        'oob_score': True
    },
    'svm': {
        'random_state': RANDOM_STATE,
        'probability': True,
        'C': 0.5,  # Reduced C for regularization
        'kernel': 'rbf',
        'gamma': 'scale',
        'class_weight': 'balanced'
    },
    'gradient_boosting': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'learning_rate': 0.05,  # Reduced learning rate
        'max_depth': 3,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'subsample': 0.8  # Use 80% of samples per tree
    },
    'xgboost': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'scale_pos_weight': 3
    }
}

# Evaluation metrics
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']