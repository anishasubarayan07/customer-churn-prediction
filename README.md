# Customer Churn Prediction (Telco) – Machine Learning

Predicts which telecom customers are likely to leave (churn) and identifies the main factors behind churn, using Python, Scikit-learn and XGBoost.

## Business problem
Keeping an existing customer is usually cheaper than winning a new one. If likely churners are identified early, a retention team can contact them with an offer before they leave.

## Dataset
IBM Telco Customer Churn sample dataset (public, from the IBM GitHub repository `IBM/telco-customer-churn-on-icp4d`): 7,043 customers, 21 columns (contract, tenure, services, billing, monthly charges). Target: `Churn` (Yes/No). File: `Telco-Customer-Churn.csv`.

## Approach
1. **Cleaning:** converted `TotalCharges` to numeric; the 11 blank values belong to customers with tenure 0 (new, not yet billed), so they were set to 0. Dropped `customerID`; encoded `Churn` as 1/0.
2. **EDA:** churn rate overall and by contract, internet service and payment method; tenure and monthly charges vs churn.
3. **Preprocessing:** one-hot encoding for categorical columns, scaling for numeric columns, stratified 80/20 train-test split.
4. **Modelling:** Logistic Regression (baseline), Random Forest, XGBoost, with class weighting for the imbalanced target and 5-fold cross-validation.
5. **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix and ROC curves.
6. **Interpretation:** feature importance and business recommendations.

## Results (hold-out test set, 1,409 customers)
| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.846 | 0.738 | 0.504 | 0.783 | 0.614 | 0.842 |
| Random Forest | 0.847 | 0.757 | 0.529 | 0.770 | 0.627 | 0.842 |
| XGBoost | 0.847 | 0.751 | 0.520 | 0.786 | 0.626 | 0.846 |

XGBoost scored highest on test ROC-AUC (0.846), but all three models performed very similarly, so the difference is small.

## Key findings
- Overall churn rate: 26.5%.
- Month-to-month contracts: 42.7% churn, compared with 11.3% (one year) and 2.8% (two year).
- Fibre optic internet: 41.9% churn, compared with 19.0% (DSL).
- Electronic check payment: 45.3% churn, compared with about 15–19% for other methods.
- Median tenure: 10 months for churned customers vs 38 months for retained customers.
- Top model features: month-to-month contract, no online security, fibre optic internet.

## Limitations and next steps
- Precision for churners is about 0.52, so roughly half of flagged customers would not actually churn. The decision threshold should be tuned to the cost of a retention offer.
- Hyperparameters were set manually; a grid or randomised search could be added.
- Next steps: SHAP explanations, threshold tuning, a simple Streamlit/Flask app for scoring a single customer.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction_GitHub.ipynb
```
Download `Telco-Customer-Churn.csv` from the IBM repository and keep it in the same folder as the notebook. If the file is missing, the notebook downloads it automatically. Then run all cells.

## Tools
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
