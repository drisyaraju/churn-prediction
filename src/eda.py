"""
Exploratory Data Analysis Module for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EDAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def dataset_overview(self):
        """Display basic dataset information"""
        print("\n" + "="*60)
        print("TASK 3: EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        
        print(f"\nFirst 5 records:")
        print(self.df.head())
        
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.any():
            print(missing[missing > 0])
        else:
            print("No missing values found")
        
        print(f"\nStatistical Summary (Numerical Features):")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())
        
        print(f"\nCategorical Features:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            print(f"{col}: {unique_values} unique values")
        
    def plot_class_distribution(self, save=True):
        """Plot target variable distribution"""
        plt.figure(figsize=(12, 5))
        
        # Pie chart
        plt.subplot(1, 2, 1)
        churn_counts = self.df[config.TARGET_COLUMN].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        plt.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
                colors=colors, startangle=90, explode=(0, 0.05))
        plt.title('Customer Churn Distribution')
        
        # Bar chart
        plt.subplot(1, 2, 2)
        ax = sns.countplot(x=config.TARGET_COLUMN, data=self.df, palette=colors)
        plt.title('Customer Count by Churn Status')
        plt.ylabel('Count')
        plt.xlabel('Churn')
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():,}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(config.PLOTS_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_numerical_features(self, save=True):
        """Plot numerical features distribution"""
        numerical_cols = config.NUMERICAL_COLUMNS
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Tenure distribution
        sns.histplot(data=self.df, x='tenure', hue=config.TARGET_COLUMN, 
                    multiple='layer', ax=axes[0, 0], palette=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Tenure Distribution by Churn')
        axes[0, 0].set_xlabel('Tenure (months)')
        
        # Monthly charges
        sns.boxplot(data=self.df, x=config.TARGET_COLUMN, y='MonthlyCharges', 
                   ax=axes[0, 1], palette=['#2ecc71', '#e74c3c'])
        axes[0, 1].set_title('Monthly Charges by Churn')
        
        # Total charges
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        sns.histplot(data=self.df, x='TotalCharges', hue=config.TARGET_COLUMN, 
                    multiple='layer', ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
        axes[1, 0].set_title('Total Charges Distribution by Churn')
        
        # Tenure vs Monthly charges scatter
        self.df['Churn_Numeric'] = self.df[config.TARGET_COLUMN].map({'Yes': 1, 'No': 0})
        scatter = axes[1, 1].scatter(self.df['tenure'], self.df['MonthlyCharges'],
                                    c=self.df['Churn_Numeric'], cmap='coolwarm', alpha=0.5)
        axes[1, 1].set_xlabel('Tenure (months)')
        axes[1, 1].set_ylabel('Monthly Charges')
        axes[1, 1].set_title('Tenure vs Monthly Charges')
        plt.colorbar(scatter, ax=axes[1, 1], label='Churn')
        
        plt.tight_layout()
        if save:
            plt.savefig(config.PLOTS_DIR / 'numerical_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_categorical_features(self, save=True):
        """Plot categorical features analysis"""
        categorical_cols = ['Contract', 'InternetService', 'PaymentMethod', 
                           'TechSupport', 'OnlineSecurity', 'MultipleLines']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, col in enumerate(categorical_cols):
            if col in self.df.columns:
                churn_rate = self.df.groupby(col)[config.TARGET_COLUMN].apply(
                    lambda x: (x == 'Yes').mean() * 100
                ).sort_values(ascending=False)
                
                ax = churn_rate.plot(kind='bar', ax=axes[idx], color='coral')
                axes[idx].set_title(f'Churn Rate by {col}', fontweight='bold')
                axes[idx].set_ylabel('Churn Rate (%)')
                axes[idx].set_xlabel(col)
                axes[idx].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for i, v in enumerate(churn_rate.values):
                    axes[idx].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(config.PLOTS_DIR / 'categorical_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_correlation_matrix(self, save=True):
        """Plot correlation matrix"""
        # Create a copy and encode categorical variables
        df_encoded = self.df.copy()
        df_encoded[config.TARGET_COLUMN] = df_encoded[config.TARGET_COLUMN].map({'Yes': 1, 'No': 0})
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
        
        if 'TotalCharges' in df_encoded.columns:
            df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
        df_encoded = df_encoded.dropna()
        
        plt.figure(figsize=(14, 12))
        correlation = df_encoded.corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        
        sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(config.PLOTS_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_full_eda(self):
        """Run complete EDA pipeline"""
        self.dataset_overview()
        self.plot_class_distribution()
        self.plot_numerical_features()
        self.plot_categorical_features()
        self.plot_correlation_matrix()