# Diabetes Risk Prediction Dashboard

Cross-country diabetes risk analysis and prediction using US BRFSS and Canadian CCHS health survey data.

## Problem

Diabetes risk models are often trained on a single national dataset; comparing US and Canadian survey data tests whether predictors and model performance hold across different healthcare systems and survey designs.

## Approach

Harmonized BRFSS 2015 (253,680 records) and CCHS (108,252 records) by aligning columns, recoding categorical variables, and engineering comparable features (BMI, blood pressure, cholesterol, smoking, physical activity). Trained Logistic Regression, Random Forest, and XGBoost on each dataset with SMOTE for class imbalance. Delivered an interactive Plotly dashboard for EDA, model comparison, and personalized risk assessment.

## Results

| Model | BRFSS (US) ROC-AUC | CCHS (Canada) ROC-AUC |
|-------|--------------------|-----------------------|
| Logistic Regression | 0.8149 | 0.7729 |
| Random Forest | 0.7913 | 0.7055 |
| XGBoost | 0.8206 | 0.7315 |

Consistent top predictors across countries: high blood pressure, BMI, and high cholesterol. Logistic Regression showed the smallest cross-country performance gap.

## Tech stack

Python, scikit-learn, XGBoost, pandas, Plotly.js (static dashboard), HTML/CSS/JavaScript

## How to run

**Browser dashboard (no install):** open the GitHub Pages URL below.

**Local analysis:** clone the repo and run notebooks or training scripts included in the repository with dependencies from `requirements.txt` if present.

## Screenshot / demo

**Live dashboard:** https://danieldemoz.github.io/Diabetes-Risk-Prediction-Using-BRFSS-US-and-CCHS-Canada/

*For educational and research purposes only—not for medical decisions.*
