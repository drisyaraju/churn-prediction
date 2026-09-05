"""
Main execution script for Customer Churn Prediction
Run this file to execute the complete pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent plot popups
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import config
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from eda import EDAnalyzer
from model_training import ModelTrainer
from evaluation import ModelEvaluator

def main():
    """Main execution function"""
    
    print("="*80)
    print("CUSTOMER CHURN PREDICTION USING MACHINE LEARNING")
    print("="*80)
    print("\nProject: IBM Telco Customer Churn Prediction")
    print("Dataset: IBM Telco Customer Churn Dataset")
    print("Source: Kaggle")
    print("="*80)
    
    # Task 1-2: Load dataset
    print("\n" + "="*60)
    print("TASK 1-2: LOADING DATASET")
    print("="*60)
    
    loader = DataLoader()
    df = loader.load_dataset()
    
    if df is None:
        print("Failed to load dataset. Please check your internet connection or Kaggle credentials.")
        return
    
    print(f"\nDataset loaded successfully!")
    print(f"Shape: {df.shape}")
    
    # Task 3: Exploratory Data Analysis
    print("\n" + "="*60)
    print("TASK 3: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    eda = EDAnalyzer(df)
    eda.run_full_eda()
    
    # Task 4: Data Preprocessing
    print("\n" + "="*60)
    print("TASK 4: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    
    # Save preprocessed data
    X_with_target = X.copy()
    X_with_target['Churn'] = y
    X_with_target.to_csv(config.DATA_DIR / 'preprocessed_data.csv', index=False)
    print(f"\n✓ Preprocessed data saved to {config.DATA_DIR / 'preprocessed_data.csv'}")
    
    # Task 5: Split dataset
    print("\n" + "="*60)
    print("TASK 5: TRAIN-TEST SPLIT")
    print("="*60)
    
    trainer = ModelTrainer(X, y)
    X_train, X_test, y_train, y_test = trainer.split_data()
    
    # Task 6: Train models
    print("\n" + "="*60)
    print("TASK 6: TRAINING CLASSIFICATION MODELS")
    print("="*60)
    
    trained_models = trainer.train_models()
    
    # Cross-validation
    cv_results = trainer.cross_validate()
    
    # Task 7: Evaluate models
    print("\n" + "="*60)
    print("TASK 7: MODEL EVALUATION")
    print("="*60)
    
    evaluator = ModelEvaluator(trained_models, X_test, y_test)
    evaluation_results = evaluator.evaluate_all_models()
    
    # Plot results (saved to files, no popups)
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    
    # Task 8: Compare models
    print("\n" + "="*60)
    print("TASK 8: MODEL COMPARISON")
    print("="*60)
    
    comparison_df = evaluator.create_comparison_table()
    
    # Task 9: Identify best model
    print("\n" + "="*60)
    print("TASK 9: BEST MODEL")
    print("="*60)
    
    best_model_name, best_metrics = evaluator.identify_best_model(comparison_df)
    
    # Task 10: Feature importance analysis
    print("\n" + "="*60)
    print("TASK 10: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Analyze feature importance for tree-based models
    for model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
        if model_name in trained_models:
            model = trained_models[model_name]
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                print(f"\n{model_name} - Top 10 Most Important Features:")
                print("-" * 50)
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
                print(f"✓ Saved feature importance plot")
    
    # Generate business insights
    print("\n" + "="*60)
    print("BUSINESS INSIGHTS")
    print("="*60)
    
    insights = [
        "1. Contract Type: Month-to-month contracts have highest churn",
        "2. Tenure: New customers (0-12 months) are at highest risk",
        "3. Internet Service: Fiber optic users show higher churn rates",
        "4. Payment Method: Electronic check users are more likely to churn",
        "5. Support Services: Lack of tech support increases churn risk"
    ]
    
    for insight in insights:
        print(f"   {insight}")
    
    # Save insights to file
    with open(config.RESULTS_DIR / 'business_insights.txt', 'w') as f:
        f.write("BUSINESS INSIGHTS\n")
        f.write("="*50 + "\n\n")
        for insight in insights:
            f.write(insight + "\n")
    print(f"\n✓ Business insights saved to {config.RESULTS_DIR / 'business_insights.txt'}")
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\nSummary:")
    print(f"1. Dataset: {df.shape[0]} customers, {X.shape[1]} features")
    print(f"2. Models trained: {len(trained_models)}")
    print(f"3. Best model: {best_model_name}")
    print(f"4. Best F1-score: {best_metrics['F1-score']:.4f}")
    print(f"5. Best Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"6. Best ROC-AUC: {best_metrics['ROC-AUC']:.4f}")
    
    print("\nOutputs saved to:")
    print(f"- Plots: {config.PLOTS_DIR}")
    print(f"- Results: {config.RESULTS_DIR}")
    print(f"- Preprocessed data: {config.DATA_DIR / 'preprocessed_data.csv'}")
    
    print("\nFiles generated:")
    print("\nPlots:")
    for plot_file in config.PLOTS_DIR.glob("*.png"):
        print(f"  - {plot_file.name}")
    
    print("\nResults:")
    for result_file in config.RESULTS_DIR.glob("*"):
        print(f"  - {result_file.name}")
    
    print("\n" + "="*80)
    print("Thank you for using the Customer Churn Prediction System!")
    print("="*80)

if __name__ == "__main__":
    main()