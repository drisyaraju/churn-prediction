"""
Quick test script to verify each component works
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor

def test_data_loading():
    print("\n" + "="*50)
    print("TESTING DATA LOADING")
    print("="*50)
    
    loader = DataLoader()
    df = loader.load_dataset()
    
    if df is not None:
        print(f"✓ Data loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns[:5])}...")
        return df
    else:
        print("✗ Data loading failed")
        return None

def test_preprocessing(df):
    print("\n" + "="*50)
    print("TESTING PREPROCESSING")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    
    print(f"✓ Preprocessing successful")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

def test_eda(df):
    print("\n" + "="*50)
    print("TESTING EDA")
    print("="*50)
    
    from eda import EDAnalyzer
    eda = EDAnalyzer(df)
    eda.dataset_overview()
    print("✓ EDA successful")

if __name__ == "__main__":
    # Test data loading
    df = test_data_loading()
    
    if df is not None:
        # Test preprocessing
        X, y = test_preprocessing(df)
        
        # Test EDA
        test_eda(df)
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*50)