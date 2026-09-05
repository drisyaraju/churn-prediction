"""
Data Preprocessing Module for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_columns = config.NUMERICAL_COLUMNS
        
    def clean_data(self, df):
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        df_cleaned = df.copy()
        print(f"Initial shape: {df_cleaned.shape}")
        
        df_cleaned = df_cleaned.drop(columns=config.DROP_COLUMNS, errors='ignore')
        
        leakage_columns = ['Churn_Numeric', 'Churn_Encoded', 'Churn_Label']
        df_cleaned = df_cleaned.drop(columns=leakage_columns, errors='ignore')
        
        print(f"Shape after removing columns: {df_cleaned.shape}")
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].replace(' ', np.nan)
        
        if 'TotalCharges' in df_cleaned.columns:
            df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
        
        missing_values = df_cleaned.isnull().sum()
        missing_total = missing_values.sum()
        print(f"\nMissing values found: {missing_total}")
        
        if missing_total > 0:
            print("Missing values by column:")
            print(missing_values[missing_values > 0])
            
            for col in df_cleaned.select_dtypes(include=[np.number]).columns:
                if col != config.TARGET_COLUMN:
                    if df_cleaned[col].isnull().sum() > 0:
                        median_value = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_value)
                        print(f"Filled {col} missing values with median: {median_value:.2f}")
            
            for col in df_cleaned.select_dtypes(include=['object']).columns:
                if col != config.TARGET_COLUMN:
                    if df_cleaned[col].isnull().sum() > 0:
                        mode_value = df_cleaned[col].mode()[0]
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                        print(f"Filled {col} missing values with mode: {mode_value}")
        
        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            df_cleaned = df_cleaned.drop_duplicates()
            print(f"\nRemoved {duplicates} duplicate rows")
        
        print(f"Final shape: {df_cleaned.shape}")
        
        remaining_nan = df_cleaned.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"WARNING: {remaining_nan} NaN values still remain!")
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().sum() > 0:
                    if df_cleaned[col].dtype in ['int64', 'float64']:
                        df_cleaned[col] = df_cleaned[col].fillna(0)
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna('Unknown')
        
        return df_cleaned
    
    def encode_features(self, df):
        print("\n" + "="*60)
        print("FEATURE ENCODING")
        print("="*60)
        
        df_encoded = df.copy()
        
        if config.TARGET_COLUMN in df_encoded.columns:
            df_encoded[config.TARGET_COLUMN] = df_encoded[config.TARGET_COLUMN].map({
                'Yes': 1, 
                'No': 0
            })
            print(f"Target variable encoded: {df_encoded[config.TARGET_COLUMN].value_counts().to_dict()}")
        
        categorical_columns = [
            col for col in df_encoded.select_dtypes(include=['object']).columns 
            if col != config.TARGET_COLUMN
        ]
        print(f"\nCategorical columns to encode: {list(categorical_columns)}")
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return df_encoded
    
    def scale_features(self, df):
        print("\n" + "="*60)
        print("FEATURE SCALING")
        print("="*60)
        
        df_scaled = df.copy()
        
        numerical_cols = [
            col for col in self.numerical_columns 
            if col in df_scaled.columns and col != config.TARGET_COLUMN
        ]
        
        if numerical_cols:
            print(f"Scaling numerical columns: {numerical_cols}")
            df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
            
            for col in numerical_cols:
                print(f"{col}: mean={df_scaled[col].mean():.2f}, std={df_scaled[col].std():.2f}")
        
        return df_scaled
    
    def prepare_data(self, df):
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        df_cleaned = self.clean_data(df)
        df_encoded = self.encode_features(df_cleaned)
        df_scaled = self.scale_features(df_encoded)
        
        if config.TARGET_COLUMN in df_scaled.columns:
            X = df_scaled.drop(columns=[config.TARGET_COLUMN])
            y = df_scaled[config.TARGET_COLUMN]
            
            target_related = [col for col in X.columns if 'churn' in col.lower()]
            if target_related:
                print(f"\nWARNING: Found target-related columns in features: {target_related}")
                X = X.drop(columns=target_related)
            
            if X.isnull().sum().sum() > 0:
                print("\nWARNING: NaN values found in features. Filling with 0...")
                X = X.fillna(0)
            
            print(f"\nFinal feature check:")
            print(f"  Target in features? {'YES - ERROR!' if config.TARGET_COLUMN in X.columns else 'No - Good'}")
            print(f"  Any churn-related columns? {'YES - ERROR!' if any('churn' in col.lower() for col in X.columns) else 'No - Good'}")
            print(f"  NaN values: {X.isnull().sum().sum()}")
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Features shape: {X.shape}")
        print(f"Feature columns ({len(X.columns)}): {list(X.columns)}")
        
        if y is not None:
            print(f"\nTarget shape: {y.shape}")
            print(f"Target distribution:")
            print(y.value_counts())
        
        return X, y