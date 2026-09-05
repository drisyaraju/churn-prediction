"""
Model Training Module with Overfitting Prevention
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import config
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Skipping XGBoost model.")

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.trained_models = {}
        self.cv_results = {}
        
    def split_data(self):
        """Split data into training and testing sets"""
        print("\n" + "="*60)
        print("TASK 5: TRAIN-TEST SPLIT")
        print("="*60)
        
        # Verify no NaN values
        if self.X.isnull().sum().sum() > 0:
            print("WARNING: NaN values found in X. Filling with 0...")
            self.X = self.X.fillna(0)
        
        # Verify target not in features
        if config.TARGET_COLUMN in self.X.columns:
            print(f"ERROR: Target column '{config.TARGET_COLUMN}' found in features!")
            self.X = self.X.drop(columns=[config.TARGET_COLUMN])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=self.y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        print(f"Number of features: {self.X_train.shape[1]}")
        print(f"\nTraining set churn rate: {self.y_train.mean()*100:.2f}%")
        print(f"Testing set churn rate: {self.y_test.mean()*100:.2f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_models(self):
        """Create model instances with regularization"""
        print("\n" + "="*60)
        print("TASK 6: TRAINING CLASSIFICATION MODELS")
        print("="*60)
        
        self.models = {
            'Logistic Regression': LogisticRegression(**config.MODEL_PARAMS['logistic_regression']),
            'Decision Tree': DecisionTreeClassifier(**config.MODEL_PARAMS['decision_tree']),
            'Random Forest': RandomForestClassifier(**config.MODEL_PARAMS['random_forest']),
            'SVM': SVC(**config.MODEL_PARAMS['svm']),
            'Gradient Boosting': GradientBoostingClassifier(**config.MODEL_PARAMS['gradient_boosting'])
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(**config.MODEL_PARAMS['xgboost'])
        
        print(f"\nModels to train:")
        for i, name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {name}")
        
        return self.models
    
    def train_models(self):
        """Train all models and check for overfitting"""
        print("\n" + "="*60)
        print("MODEL TRAINING PROGRESS")
        print("="*60)
        
        self.create_models()
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                
                # Calculate training accuracy
                train_score = model.score(self.X_train, self.y_train)
                
                # Calculate cross-validation score
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                
                print(f"✓ {name} trained successfully")
                print(f"  Training accuracy: {train_score:.4f}")
                print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
                # Check for overfitting
                if train_score > 0.95:
                    print(f"  ⚠️ WARNING: Very high training accuracy - possible overfitting!")
                elif train_score - cv_scores.mean() > 0.1:
                    print(f"  ⚠️ WARNING: Gap between train and CV: {train_score - cv_scores.mean():.4f}")
                
            except Exception as e:
                print(f"✗ {name} training failed: {str(e)}")
        
        return self.trained_models