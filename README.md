
# Customer Churn Prediction

Machine learning project to predict customer churn in a telecommunications company using demographic, service, and billing data.

## 📊 Dataset

**IBM Telco Customer Churn Dataset** from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers, 21 features
- Target: Churn (Yes/No)
- Class distribution: 73% No Churn, 27% Churn

## 🎯 Results

### Best Model: XGBoost

| Metric | Score |
|--------|-------|
| Accuracy | 0.7473 |
| F1-Score | 0.6227 |
| ROC-AUC | 0.8427 |

### Model Comparison

| Algorithm | Accuracy | F1-Score | ROC-AUC |
|-----------|----------|----------|---------|
| **XGBoost** | **0.7473** | **0.6227** | **0.8427** |
| Gradient Boosting | 0.8203 | 0.5987 | 0.8456 |
| Random Forest | 0.7974 | 0.5789 | 0.8234 |
| Decision Tree | 0.7635 | 0.5523 | 0.7468 |
| Logistic Regression | 0.7480 | 0.5345 | 0.7441 |
| SVM | 0.7539 | 0.5210 | 0.7398 |

## 🔍 Key Features Affecting Churn

1. Monthly Charges
2. Online Security
3. Total Charges
4. Tech Support
5. Payment Method

## 📁 Project Structure

```
churn_prediction/
├── data/
│   └── Telco-Customer-Churn.csv
├── outputs/
│   ├── plots/          # Visualizations
│   └── results/        # Model results
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── run_final.py    # Main pipeline
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
git clone https://github.com/drisyaraju/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
```

## 💻 Usage

```bash
cd src
python run_final.py
```

## 🛠️ Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC


## 👤 Author

**Drisya Raju**
- GitHub: [@drisyaraju](https://github.com/drisyaraju)
