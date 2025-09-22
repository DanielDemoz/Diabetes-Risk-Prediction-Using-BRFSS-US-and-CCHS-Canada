README — Diabetes Risk Prediction using BRFSS & CCHS

Author: Daniel S. Demoz

📖 Project Overview

This project integrates two large health survey datasets—the U.S. BRFSS (Behavioral Risk Factor Surveillance System) and the Canadian CCHS (Canadian Community Health Survey)—to analyze risk factors for diabetes. It involves:

Data loading, cleaning, and preprocessing.

Exploratory Data Analysis (EDA).

Building predictive models (Logistic Regression, Random Forest, and XGBoost).

Model comparison across both datasets.

Feature importance interpretation using SHAP and tree-based models.

The work highlights the similarities and differences between U.S. and Canadian health data, emphasizing BMI, smoking, physical activity, and diet as key diabetes-related factors.

⚙️ Libraries Used

The project uses the following Python libraries:

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning:

scikit-learn (train/test split, scaling, logistic regression, random forest, metrics)

xgboost (gradient boosting)

Imbalanced Data Handling: imblearn.SMOTE

Interpretability: shap

📂 Datasets

BRFSS (U.S.): diabetes_012_health_indicators_BRFSS2015.csv

Health indicators such as BMI, smoking, physical activity, blood pressure, and cholesterol.

CCHS (Canada): pumf_cchs.csv

Self-reported and adjusted BMI, smoking habits, physical activity (minutes & WHO guidelines), fruit/vegetable intake, and chronic disease indicators.

📌 Both datasets are assumed to be stored in Google Drive and accessed from Colab.

🔄 Data Preprocessing
BRFSS

Renamed columns for clarity (e.g., Diabetes_012 → diabetes, HighBP → high_bp).

Selected relevant health indicators for analysis.

CCHS

Renamed variables based on data dictionary.

Recoded categorical values into binary (0/1) where applicable (diabetes, high_bp, high_chol).

Replaced special codes (e.g., 7=Refusal, 9=Not Stated) with NaN.

Imputed missing values:

Numerical columns → Median.

Categorical-like columns → Mode.

Binary columns → Mode.

Applied one-hot encoding to categorical variables (smoker_type, smoker_status, phys_activity_who).

📊 Exploratory Data Analysis (EDA)
Key Visualizations

Histograms: BMI, fruit/vegetable intake, physical activity distributions.

Boxplots: BMI/fruit/veg vs. diabetes outcomes.

Countplots: High BP, cholesterol, smoking distributions.

Correlation Heatmaps: Diabetes associations with other variables.

Findings

BMI is consistently higher among individuals with diabetes in both datasets.

Physical activity tends to be lower in diabetic groups, especially in CCHS.

High blood pressure and cholesterol are strong diabetes risk indicators across both datasets.

Smoking status is defined differently across datasets (binary vs. multi-category).

🤖 Machine Learning Models

Models were trained separately for CCHS and compared to BRFSS:

Logistic Regression

Baseline interpretable model.

Performance measured by ROC-AUC.

Random Forest Classifier

Ensemble model for non-linear relationships.

Feature importance used for interpretation.

XGBoost Classifier

Gradient boosting model, often outperforming others in structured data.

Feature importance + SHAP values used for interpretation.

Class Imbalance Handling

Applied SMOTE on the training data to balance diabetes vs. non-diabetes cases.

📈 Results

ROC-AUC Scores (CCHS models):

Logistic Regression → ~0.70 (baseline performance).

Random Forest → ~0.75.

XGBoost → ~0.77 (best performance).

Feature Importances (CCHS & BRFSS comparison):

Top predictors across both datasets:

BMI (measured or self-reported).

High blood pressure.

High cholesterol.

Differences exist in smoking and physical activity due to dataset definitions.

🔍 Model Interpretability

Random Forest & XGBoost:

Feature importance plots highlight BMI, high blood pressure, and cholesterol as top contributors.

SHAP values:

Show feature-level impacts on individual predictions.

Confirms BMI and hypertension as strong diabetes predictors.

📌 Key Takeaways

Consistent risk factors: Both U.S. and Canadian datasets reinforce the strong link between BMI, high blood pressure, cholesterol, and diabetes.

Differences in variables: CCHS provides richer categorical detail (e.g., smoker types, activity levels), while BRFSS uses simpler binary coding.

Data harmonization challenge: Direct cross-country comparison requires careful handling of differences in survey methodology and coding.

Model performance: XGBoost consistently outperforms Logistic Regression and Random Forest.
