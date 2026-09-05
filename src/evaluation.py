"""
Model Evaluation Module for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import config
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, trained_models, X_test, y_test):
        self.trained_models = trained_models
        self.X_test = X_test
        self.y_test = y_test
        self.evaluation_results = {}
        
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("TASK 7: MODEL EVALUATION")
        print("="*60)
        
        for name, model in self.trained_models.items():
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Get probability predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(self.X_test)
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1-score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
                'Confusion Matrix': confusion_matrix(self.y_test, y_pred),
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            }
            
            self.evaluation_results[name] = metrics
            
            # Print detailed results
            print(f"\n{'='*50}")
            print(f"Model: {name}")
            print(f"{'='*50}")
            print(f"Accuracy: {metrics['Accuracy']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall: {metrics['Recall']:.4f}")
            print(f"F1-score: {metrics['F1-score']:.4f}")
            print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
            
            print("\nConfusion Matrix:")
            print(metrics['Confusion Matrix'])
            
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        return self.evaluation_results
    
    def plot_confusion_matrices(self, save=True):
        """Plot confusion matrices for all models"""
        n_models = len(self.evaluation_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()
        
        for idx, (name, results) in enumerate(self.evaluation_results.items()):
            if idx < len(axes):
                sns.heatmap(results['Confusion Matrix'], annot=True, fmt='d', 
                           cmap='Blues', ax=axes[idx], cbar=False)
                axes[idx].set_title(f'{name}', fontweight='bold')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
                axes[idx].set_xticklabels(['No Churn', 'Churn'])
                axes[idx].set_yticklabels(['No Churn', 'Churn'])
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save:
            plt.savefig(config.PLOTS_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, save=True):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, results in self.evaluation_results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['Probabilities'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(config.PLOTS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_table(self, save=True):
        """Create model comparison table"""
        comparison_data = []
        
        for name, results in self.evaluation_results.items():
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
        
        print("\n" + "="*80)
        print("TASK 8: MODEL PERFORMANCE COMPARISON TABLE")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        if save:
            comparison_df.to_csv(config.RESULTS_DIR / 'model_comparison.csv', index=False)
            print(f"\nComparison table saved to {config.RESULTS_DIR / 'model_comparison.csv'}")
        
        return comparison_df
    
    def identify_best_model(self, comparison_df):
        """Identify best performing model"""
        print("\n" + "="*60)
        print("TASK 9: BEST MODEL IDENTIFICATION")
        print("="*60)
        
        # Find best models by different metrics
        best_by_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Algorithm']
        best_by_f1 = comparison_df.loc[comparison_df['F1-score'].idxmax(), 'Algorithm']
        best_by_auc = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Algorithm']
        best_by_recall = comparison_df.loc[comparison_df['Recall'].idxmax(), 'Algorithm']
        
        print(f"\nBest by Accuracy: {best_by_accuracy}")
        print(f"Best by F1-score: {best_by_f1}")
        print(f"Best by ROC-AUC: {best_by_auc}")
        print(f"Best by Recall: {best_by_recall}")
        
        # For churn prediction, prioritize F1-score
        best_model = best_by_f1
        best_metrics = comparison_df[comparison_df['Algorithm'] == best_model].iloc[0]
        
        print(f"\n{'='*50}")
        print(f"RECOMMENDED MODEL: {best_model}")
        print(f"{'='*50}")
        print(f"\nJustification:")
        print(f"1. Highest F1-score ({best_metrics['F1-score']:.4f})")
        print(f"2. Balanced performance:")
        print(f"   - Precision: {best_metrics['Precision']:.4f}")
        print(f"   - Recall: {best_metrics['Recall']:.4f}")
        print(f"3. Strong ROC-AUC ({best_metrics['ROC-AUC']:.4f})")
        print(f"\nThis model provides the optimal balance between:")
        print(f"- Identifying customers likely to churn (recall)")
        print(f"- Minimizing false alarms (precision)")
        
        return best_model, best_metrics

# Add main execution block
if __name__ == "__main__":
    print("Model Evaluation Module")
    print("This module should be imported and used through main.py")
    print("Run: python main.py")