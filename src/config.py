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

# Model hyperparameters
MODEL_PARAMS = {
    'logistic_regression': {
        'random_state': RANDOM_STATE,
        'max_iter': 1000,
        'C': 1.0,
        'class_weight': 'balanced'
    },
    'decision_tree': {
        'random_state': RANDOM_STATE,
        'max_depth': 5,
        'min_samples_split': 10,
        'class_weight': 'balanced'
    },
    'random_forest': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'class_weight': 'balanced'
    },
    'svm': {
        'random_state': RANDOM_STATE,
        'probability': True,
        'C': 1.0,
        'kernel': 'rbf',
        'class_weight': 'balanced'
    },
    'gradient_boosting': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    },
    'xgboost': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'scale_pos_weight': 3
    }
}

# Evaluation metrics
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']