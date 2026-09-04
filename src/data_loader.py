"""
Data Loader Module - Loads dataset using Kaggle API or direct URL
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import io
import config
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self):
        self.dataset_path = config.DATASET_PATH
        self.data_dir = config.DATA_DIR
        
    def load_from_kaggle_api(self):
        """Load dataset directly using Kaggle API"""
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            print("Attempting to load data using Kaggle API...")
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            print(f"Downloading dataset: {config.KAGGLE_DATASET}")
            api.dataset_download_files(
                config.KAGGLE_DATASET,
                path=str(self.data_dir),
                unzip=True
            )
            
            # Load the dataset
            df = pd.read_csv(self.dataset_path)
            print(f"✓ Dataset loaded successfully from Kaggle API")
            
            return df
            
        except Exception as e:
            print(f"Kaggle API loading failed: {str(e)}")
            print("Falling back to alternative methods...")
            return None
    
    def load_from_github(self):
        """Load dataset from GitHub raw URL"""
        try:
            print(f"Loading dataset from GitHub...")
            response = requests.get(config.GITHUB_URL)
            
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                print(f"✓ Dataset loaded successfully from GitHub")
                
                # Save locally for future use
                df.to_csv(self.dataset_path, index=False)
                print(f"✓ Dataset saved to {self.dataset_path}")
                
                return df
            else:
                print(f"GitHub loading failed with status code: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"GitHub loading failed: {str(e)}")
            return None
    
    def load_from_opendatasets(self):
        """Load dataset using opendatasets library"""
        try:
            import opendatasets as od
            
            print("Loading dataset using opendatasets...")
            dataset_url = f"https://www.kaggle.com/datasets/{config.KAGGLE_DATASET}"
            
            # Download dataset
            od.download(dataset_url, data_dir=str(self.data_dir))
            
            # Find the CSV file
            csv_files = list(self.data_dir.rglob("*.csv"))
            
            if csv_files:
                df = pd.read_csv(csv_files[0])
                print(f"✓ Dataset loaded successfully using opendatasets")
                return df
            else:
                print("No CSV file found in downloaded dataset")
                return None
                
        except Exception as e:
            print(f"opendatasets loading failed: {str(e)}")
            return None
    
    def load_dataset(self):
        """Main method to load dataset with multiple fallback options"""
        print("\n" + "="*60)
        print("LOADING DATASET")
        print("="*60)
        
        # Check if dataset already exists locally
        if self.dataset_path.exists():
            print(f"Dataset found locally at {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            print(f"✓ Dataset loaded successfully")
            return df
        
        # Method 1: Try Kaggle API
        df = self.load_from_kaggle_api()
        if df is not None:
            return df
        
        # Method 2: Try GitHub URL
        df = self.load_from_github()
        if df is not None:
            return df
        
        # Method 3: Try opendatasets
        df = self.load_from_opendatasets()
        if df is not None:
            return df
        
        # If all methods fail
        raise Exception("""
        Failed to load dataset using all methods.
        
        Please choose one of the following options:
        1. Set up Kaggle API credentials:
           - Go to https://www.kaggle.com/settings/account
           - Click 'Create New API Token'
           - Save kaggle.json to ~/.kaggle/kaggle.json
           
        2. Manually download the dataset:
           - Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
           - Download and save as 'Telco-Customer-Churn.csv' in the data folder
           
        3. Use a direct URL to load the dataset
        """)