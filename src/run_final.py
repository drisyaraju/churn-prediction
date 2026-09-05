"""
Complete Customer Churn Prediction Pipeline
Run this file to get all results
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent plot popups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

def main():
    print("="*80)
    print("CUSTOMER CHURN PREDICTION USING MACHINE LEARNING")
    print("="*80)
    
    # Load data
    print("\n1. Loading dataset...")
    loader = DataLoader()
    df = loader.load_dataset()
    print(f"   Dataset shape: {df.shape}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Split data
    print("\n3. Splitting data...")
    trainer = ModelTrainer(X, y)
    X_train, X_test, y_train, y_test = trainer.split_data()
    
    # Train models
    print("\n4. Training models...")
    trained_models = trainer.train_models()
    
    # Evaluate models
    print("\n5. Evaluating models...")
    evaluation_results = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
        
        evaluation_results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Probabilities': y_pred_proba
        }
        
        print(f"\n   {name}:")
        print(f"     Accuracy: {evaluation_results[name]['Accuracy']:.4f}")
        print(f"     Precision: {evaluation_results[name]['Precision']:.4f}")
        print(f"     Recall: {evaluation_results[name]['Recall']:.4f}")
        print(f"     F1-score: {evaluation_results[name]['F1-score']:.4f}")
        print(f"     ROC-AUC: {evaluation_results[name]['ROC-AUC']:.4f}")
    
    # Create comparison table
    print("\n6. Model Comparison Table:")
    print("="*80)
    comparison_data = []
    for name, results in evaluation_results.items():
        comparison_data.append({
            'Algorithm': name,
            'Accuracy': results['Accuracy'],
            'Precision': results['Precision'],
            'Recall': results['Recall'],
            'F1-score': results['F1-score'],
            'ROC-AUC': results['ROC-AUC']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    comparison_df = comparison_df.sort_values('F1-score', ascending=False)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Save comparison table
    comparison_df.to_csv(config.RESULTS_DIR / 'model_comparison.csv', index=False)
    print(f"\n   ✓ Comparison table saved to {config.RESULTS_DIR / 'model_comparison.csv'}")
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Algorithm']
    best_metrics = comparison_df.iloc[0]
    
    print("\n7. Best Model:")
    print("="*50)
    print(f"   Model: {best_model_name}")
    print(f"   Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"   F1-score: {best_metrics['F1-score']:.4f}")
    print(f"   ROC-AUC: {best_metrics['ROC-AUC']:.4f}")
    print("="*50)
    
    # Plot confusion matrices
    print("\n8. Saving confusion matrices...")
    n_models = len(evaluation_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.ravel()
    
    for idx, (name, results) in enumerate(evaluation_results.items()):
        if idx < len(axes):
            sns.heatmap(results['Confusion Matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticklabels(['No Churn', 'Churn'])
            axes[idx].set_yticklabels(['No Churn', 'Churn'])
    
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {config.PLOTS_DIR / 'confusion_matrices.png'}")
    
    # Plot ROC curves
    print("\n9. Saving ROC curves...")
    plt.figure(figsize=(10, 8))
    
    for name, results in evaluation_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['Probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {config.PLOTS_DIR / 'roc_curves.png'}")
    
    # Feature importance
    print("\n10. Feature Importance Analysis:")
    print("="*60)
    
    for model_name in ['Random Forest', 'Gradient Boosting']:
        if model_name in trained_models:
            model = trained_models[model_name]
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                print(f"\n   {model_name} - Top 10 Features:")
                print(feature_importance.head(10).to_string(index=False))
                
                # Save feature importance
                feature_importance.to_csv(
                    config.RESULTS_DIR / f'feature_importance_{model_name.lower().replace(" ", "_")}.csv',
                    index=False
                )
                
                # Plot feature importance
                plt.figure(figsize=(10, 8))
                top_features = feature_importance.head(10)
                plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1])
                plt.xlabel('Importance')
                plt.title(f'Top 10 Feature Importance ({model_name})')
                plt.tight_layout()
                plt.savefig(
                    config.PLOTS_DIR / f'feature_importance_{model_name.lower().replace(" ", "_")}.png',
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
                print(f"   ✓ Saved feature importance for {model_name}")
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\nSummary:")
    print(f"   1. Dataset: {df.shape[0]} customers, {X.shape[1]} features")
    print(f"   2. Models trained: {len(trained_models)}")
    print(f"   3. Best model: {best_model_name}")
    print(f"   4. Best F1-score: {best_metrics['F1-score']:.4f}")
    print(f"   5. Best Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"   6. Best ROC-AUC: {best_metrics['ROC-AUC']:.4f}")
    
    print("\nOutputs saved to:")
    print(f"   - Plots: {config.PLOTS_DIR}")
    print(f"   - Results: {config.RESULTS_DIR}")
    
    print("\nFiles generated:")
    for plot_file in config.PLOTS_DIR.glob("*.png"):
        print(f"   - {plot_file.name}")
    for result_file in config.RESULTS_DIR.glob("*.csv"):
        print(f"   - {result_file.name}")
    
    print("\n" + "="*80)
    print("Thank you for using the Customer Churn Prediction System!")
    print("="*80)

if __name__ == "__main__":
    main()