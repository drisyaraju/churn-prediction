"""
Model Training Module for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=self.y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]} samples ({1-config.TEST_SIZE:.0%})")
        print(f"Testing set size: {self.X_test.shape[0]} samples ({config.TEST_SIZE:.0%})")
        print(f"\nTraining set churn rate: {self.y_train.mean()*100:.2f}%")
        print(f"Testing set churn rate: {self.y_test.mean()*100:.2f}%")
        
        # Check class distribution
        print(f"\nTraining set class distribution:")
        print(f"  No Churn (0): {(self.y_train == 0).sum()}")
        print(f"  Churn (1): {(self.y_train == 1).sum()}")
        
        print(f"\nTesting set class distribution:")
        print(f"  No Churn (0): {(self.y_test == 0).sum()}")
        print(f"  Churn (1): {(self.y_test == 1).sum()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_models(self):
        """Create model instances"""
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
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {model_name}")
        
        return self.models
    
    def train_models(self):
        """Train all models"""
        print("\n" + "="*60)
        print("MODEL TRAINING PROGRESS")
        print("="*60)
        
        self.create_models()
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            
            # Calculate training accuracy
            train_score = model.score(self.X_train, self.y_train)
            print(f"✓ {name} trained successfully")
            print(f"  Training accuracy: {train_score:.4f}")
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("="*60)
        
        return self.trained_models
    
    def cross_validate(self, cv=5):
        """Perform cross-validation"""
        print("\n" + "="*60)
        print("CROSS-VALIDATION (5-fold)")
        print("="*60)
        
        for name, model in self.trained_models.items():
            try:
                scores = cross_val_score(model, self.X_train, self.y_train, 
                                        cv=cv, scoring='roc_auc')
                self.cv_results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
                print(f"\n{name}:")
                print(f"  Mean ROC-AUC: {scores.mean():.4f}")
                print(f"  Std Dev: {scores.std():.4f}")
                print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
            except Exception as e:
                print(f"\n{name}: Cross-validation failed - {str(e)}")
        
        return self.cv_results