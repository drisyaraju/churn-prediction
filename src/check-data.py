"""
Quick script to check data for leakage and NaN values
"""

import pandas as pd
import numpy as np
import config
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor

def check_data():
    print("="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    # Load data
    loader = DataLoader()
    df = loader.load_dataset()
    
    # Check original data
    print("\n1. ORIGINAL DATA CHECK")
    print(f"   Shape: {df.shape}")
    print(f"   Target column present: {config.TARGET_COLUMN in df.columns}")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    
    print("\n2. PREPROCESSED DATA CHECK")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Target in X: {'YES - LEAKAGE!' if config.TARGET_COLUMN in X.columns else 'No - Good'}")
    print(f"   NaN in X: {X.isnull().sum().sum()}")
    print(f"   NaN in y: {y.isnull().sum()}")
    
    # Check for highly correlated features
    print("\n3. CORRELATION CHECK")
    correlation = X.corr()
    high_corr = []
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            if abs(correlation.iloc[i, j]) > 0.95:
                high_corr.append((correlation.columns[i], correlation.columns[j], correlation.iloc[i, j]))
    
    if high_corr:
        print("   Highly correlated features (>0.95):")
        for feat1, feat2, corr in high_corr:
            print(f"     {feat1} - {feat2}: {corr:.3f}")
    else:
        print("   No highly correlated features found")
    
    print("\n4. FEATURE LIST")
    print(f"   Features ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"     {i}. {col}")
    
    return X, y

if __name__ == "__main__":
    X, y = check_data()