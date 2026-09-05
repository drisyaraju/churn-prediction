"""
Quick script to check and fix overfitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

def check_overfitting():
    print("="*60)
    print("OVERFITTING CHECK")
    print("="*60)
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_dataset()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    
    # Split data
    trainer = ModelTrainer(X, y)
    X_train, X_test, y_train, y_test = trainer.split_data()
    
    # Train models
    trained_models = trainer.train_models()
    
    # Compare training and testing accuracy
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    
    results = []
    for name, model in trained_models.items():
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc
        
        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Gap': gap,
            'Overfitting': 'Yes' if gap > 0.1 else 'No'
        })
        
        print(f"\n{name}:")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Testing Accuracy: {test_acc:.4f}")
        print(f"  Gap: {gap:.4f}")
        print(f"  Overfitting: {'⚠️ YES' if gap > 0.1 else '✓ No'}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary_df))
    width = 0.35
    
    ax.bar(x - width/2, summary_df['Train Accuracy'], width, label='Training', color='blue')
    ax.bar(x + width/2, summary_df['Test Accuracy'], width, label='Testing', color='green')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training vs Testing Accuracy (Overfitting Check)')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target')
    
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'overfitting_check.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_df

if __name__ == "__main__":
    summary = check_overfitting()