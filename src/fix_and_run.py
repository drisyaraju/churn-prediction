"""
Quick fix script to remove data leakage and run the pipeline
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor

def check_and_fix():
    # Load data
    loader = DataLoader()
    df = loader.load_dataset()
    
    print("Original columns:", list(df.columns))
    
    # Check for any churn-related columns that shouldn't be there
    if 'Churn_Numeric' in df.columns:
        print("Found Churn_Numeric column - removing...")
        df = df.drop(columns=['Churn_Numeric'])
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    
    print("\nFinal features:", list(X.columns))
    print(f"Number of features: {X.shape[1]}")
    
    # Verify no leakage
    churn_cols = [col for col in X.columns if 'churn' in col.lower()]
    if churn_cols:
        print(f"ERROR: Churn columns still present: {churn_cols}")
    else:
        print("✓ No data leakage detected")
    
    return X, y

if __name__ == "__main__":
    X, y = check_and_fix()