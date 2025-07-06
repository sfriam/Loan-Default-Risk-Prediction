
# 💸 Loan Default Risk Prediction - End-to-End ML Project

## 📌 Objective
Build a machine learning model to **predict loan default risk** using LendingClub data (2007–2018). The goal is to help financial institutions make **informed lending decisions** by identifying high-risk borrowers before disbursing loans.

## 🧰 Tools & Technologies Used
- **Python (Pandas, NumPy, Matplotlib, Seaborn)**
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Random Forest
- **Hyperparameter Optimization:** Optuna
- **Model Interpretability:** SHAP
- **Environment:** Google Colab

## 🔍 Dataset Overview
- Dataset Source: LendingClub via [Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/loan-default-prediction-dataset)
- Raw dataset: `accepted_2007_to_2018Q4.csv` (~2 million records)
- Target variable: `loan_status` → binary (`Fully Paid` → 0, `Charged Off` → 1)


## 🔬 Data Preprocessing & Feature Engineering
- Selected relevant features (loan amount, term, interest rate, income, credit history, etc.)
- Encoded categorical variables using Label Encoding
- Converted % fields (like `int_rate`, `revol_util`) to float
- Handled missing values using median imputation
- Added derived features like `credit_history_length` (from issue date and earliest credit line)

## 📊 Exploratory Data Analysis (EDA)
- Analyzed class imbalance: ~20% of loans are defaults
- Visualized default rates by:
  - Grade & Sub-grade
  - Employment Length
  - Home Ownership
  - Loan Purpose
- Observed key default patterns (e.g., higher interest & lower grades → higher default)

##  Modeling Approaches
Three tree-based models were trained and compared:

| Model           | AUC-ROC | Accuracy | Recall (Default) | F1-Score (Default) |
|----------------|---------|----------|------------------|--------------------|
| Random Forest  | 0.738   | 0.68     | 0.66             | 0.45               |
| XGBoost        | 0.741   | 0.68     | 0.67             | 0.46               |
| LightGBM       | 0.743   | 0.68     | 0.68             | 0.46               |

✅ **LightGBM** was selected as the best performing model (highest AUC, best recall for default class).


## 🧪 Hyperparameter Tuning (Optuna)
- Used Optuna to maximize AUC by optimizing:
  - `learning_rate`, `num_leaves`, `max_depth`
  - `min_child_samples`, `subsample`, `colsample_bytree`
- Applied early stopping during training
- Best AUC achieved via Optuna: **0.7379**


## 📈 Threshold Tuning (Precision vs Recall)
- Evaluated model across thresholds (0.0 to 1.0)
- Identified best threshold: **0.23**
- Final classification metrics at optimal threshold:

```text
Precision (default): 0.37
Recall (default):    0.60
F1-score (default):  0.46
Accuracy:            0.72
AUC-ROC:             0.7419





