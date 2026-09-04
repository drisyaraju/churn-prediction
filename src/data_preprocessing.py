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
        """Clean the dataset by handling missing values and duplicates"""
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        df_cleaned = df.copy()
        
        # Display initial info
        print(f"Initial shape: {df_cleaned.shape}")
        
        # Remove unnecessary columns
        df_cleaned = df_cleaned.drop(columns=config.DROP_COLUMNS, errors='ignore')
        print(f"Shape after removing columns: {df_cleaned.shape}")
        
        # Replace empty strings with NaN
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].replace(' ', np.nan)
        
        # Check for missing values
        missing_values = df_cleaned.isnull().sum()
        missing_total = missing_values.sum()
        print(f"\nMissing values found: {missing_total}")
        
        if missing_total > 0:
            print("Missing values by column:")
            print(missing_values[missing_values > 0])
            
            # Convert TotalCharges to numeric
            if 'TotalCharges' in df_cleaned.columns:
                df_cleaned['TotalCharges'] = pd.to_numeric(
                    df_cleaned['TotalCharges'], 
                    errors='coerce'
                )
                # Fill missing TotalCharges with median
                median_value = df_cleaned['TotalCharges'].median()
                df_cleaned['TotalCharges'] = df_cleaned['TotalCharges'].fillna(median_value)
                print(f"Filled TotalCharges missing values with median: {median_value:.2f}")
            
            # Fill categorical missing values with mode
            for col in df_cleaned.select_dtypes(include=['object']).columns:
                if df_cleaned[col].isnull().sum() > 0:
                    mode_value = df_cleaned[col].mode()[0]
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                    print(f"Filled {col} missing values with mode: {mode_value}")
        else:
            # Convert TotalCharges to numeric if needed
            if 'TotalCharges' in df_cleaned.columns:
                df_cleaned['TotalCharges'] = pd.to_numeric(
                    df_cleaned['TotalCharges'], 
                    errors='coerce'
                )
        
        # Remove duplicates
        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            df_cleaned = df_cleaned.drop_duplicates()
            print(f"\nRemoved {duplicates} duplicate rows")
        else:
            print("\nNo duplicate rows found")
        
        print(f"Final shape: {df_cleaned.shape}")
        return df_cleaned
    
    def encode_features(self, df):
        """Encode categorical variables and target variable"""
        print("\n" + "="*60)
        print("FEATURE ENCODING")
        print("="*60)
        
        df_encoded = df.copy()
        
        # Encode target variable
        if config.TARGET_COLUMN in df_encoded.columns:
            df_encoded[config.TARGET_COLUMN] = df_encoded[config.TARGET_COLUMN].map({
                'Yes': 1, 
                'No': 0
            })
            print(f"Target variable encoded: {df_encoded[config.TARGET_COLUMN].value_counts().to_dict()}")
        
        # Identify categorical columns
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        print(f"\nCategorical columns to encode: {list(categorical_columns)}")
        
        # Encode categorical features
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return df_encoded
    
    def scale_features(self, df):
        """Scale numerical features using StandardScaler"""
        print("\n" + "="*60)
        print("FEATURE SCALING")
        print("="*60)
        
        df_scaled = df.copy()
        
        # Scale numerical columns
        numerical_cols = [col for col in self.numerical_columns if col in df_scaled.columns]
        
        if numerical_cols:
            print(f"Scaling numerical columns: {numerical_cols}")
            df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
            
            # Display scaling statistics
            for col in numerical_cols:
                print(f"{col}: mean={df_scaled[col].mean():.2f}, std={df_scaled[col].std():.2f}")
        
        return df_scaled
    
    def prepare_data(self, df):
        """Complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Clean data
        df_cleaned = self.clean_data(df)
        
        # Encode features
        df_encoded = self.encode_features(df_cleaned)
        
        # Scale features
        df_scaled = self.scale_features(df_encoded)
        
        # Separate features and target
        if config.TARGET_COLUMN in df_scaled.columns:
            X = df_scaled.drop(columns=[config.TARGET_COLUMN])
            y = df_scaled[config.TARGET_COLUMN]
        else:
            X = df_scaled
            y = None
            print("Warning: Target column not found in dataset")
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Features shape: {X.shape}")
        if y is not None:
            print(f"Target shape: {y.shape}")
            print(f"Target distribution:")
            print(y.value_counts())
        
        return X, y
    
    def preprocess_new_data(self, df):
        """Preprocess new data for prediction"""
        df_processed = self.clean_data(df)
        df_processed = self.encode_features(df_processed)
        df_processed = self.scale_features(df_processed)
        return df_processed