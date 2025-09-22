# Diabetes Risk Factor Analysis: U.S. vs. Canada  
**Author:** Daniel S. Demoz

## Project Overview  
This project integrates two large health survey datasets—the U.S. BRFSS (Behavioral Risk Factor Surveillance System) and the Canadian CCHS (Canadian Community Health Survey)—to analyze risk factors for diabetes. It includes:

- Data loading, cleaning, and preprocessing  
- Exploratory Data Analysis (EDA)  
- Predictive modeling using Logistic Regression, Random Forest, and XGBoost  
- Model comparison across both datasets  
- Feature importance interpretation using SHAP and tree-based models  

The analysis highlights key diabetes-related factors such as BMI, smoking, physical activity, and diet, while comparing health trends between the U.S. and Canada.

## Libraries Used  

**Data Handling**  
- pandas  
- numpy  

**Visualization**  
- matplotlib  
- seaborn  

**Machine Learning**  
- scikit-learn (train/test split, scaling, logistic regression, random forest, metrics)  
- xgboost (gradient boosting)  

**Imbalanced Data Handling**  
- imblearn.SMOTE  

**Interpretability**  
- shap  

## Datasets  

**BRFSS (U.S.):**  
Filename: `diabetes_012_health_indicators_BRFSS2015.csv`  
- Contains health indicators such as BMI, smoking, physical activity, blood pressure, and cholesterol  
- Source:  
  - [GitHub](https://github.com/clairerebello/Diabetes_Dataset/blob/main/diabetes_012_health_indicators_BRFSS2015.csv)  
  - [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
  - [Hugging Face](https://huggingface.co/spaces/cnasa/Diabetes_Model/blob/main/diabetes_012__health_indicators_BRFSS2015.csv)

**CCHS (Canada):**  
Filename: `pumf_cchs.csv`  
- Includes self-reported and adjusted BMI, smoking habits, physical activity (minutes & WHO guidelines), fruit/vegetable intake, and chronic disease indicators  
- Source:  
  - [Statistics Canada – Public Use Microdata File](https://www150.statcan.gc.ca/n1/pub/82m0013x/82m0013x2024001-eng.htm)  
  - [Health Data Nexus](https://healthdatanexus.ai/content/cchs-pumf/1.1/)  
  - [GitHub project using CCHS](https://github.com/venkateshsoundar/canadian-qol-analysis)

Note: Both datasets are assumed to be stored in Google Drive and accessed via Google Colab.

## Data Preprocessing  

**BRFSS**  
- Renamed columns for clarity (e.g., `Diabetes_012` → `diabetes`, `HighBP` → `high_bp`)  
- Selected relevant health indicators for analysis  

**CCHS**  
- Renamed variables based on data dictionary  
- Recoded categorical values into binary (0/1) where applicable (`diabetes`, `high_bp`, `high_chol`)  
- Replaced special codes (e.g., 7 = Refusal, 9 = Not Stated) with NaN  
- Imputed missing values:  
  - Numerical → Median  
  - Categorical-like → Mode  
  - Binary → Mode  
- Applied one-hot encoding to categorical variables (`smoker_type`, `smoker_status`, `phys_activity_who`)  

## Exploratory Data Analysis (EDA)  

**Visualizations**  
- Histograms: BMI, fruit/vegetable intake, physical activity  
- Boxplots: BMI/fruit/veg vs. diabetes outcomes  
- Countplots: High BP, cholesterol, smoking distributions  
- Correlation Heatmaps: Diabetes associations with other variables  

**Findings**  
- BMI is consistently higher among individuals with diabetes in both datasets  
- Physical activity tends to be lower in diabetic groups, especially in CCHS  
- High blood pressure and cholesterol are strong diabetes risk indicators  
- Smoking status is defined differently across datasets (binary vs. multi-category)  

## Machine Learning Models  

Models were trained separately for CCHS and compared to BRFSS:

**Logistic Regression**  
- Baseline interpretable model  
- Performance measured by ROC-AUC  

**Random Forest Classifier**  
- Ensemble model for non-linear relationships  
- Feature importance used for interpretation  

**XGBoost Classifier**  
- Gradient boosting model, often outperforming others  
- Feature importance and SHAP values used for interpretation  

**Class Imbalance Handling**  
- Applied SMOTE to balance diabetes vs. non-diabetes cases  

## Results  

**ROC-AUC Scores (CCHS Models)**  
| Model              | ROC-AUC |
|--------------------|---------|
| Logistic Regression| ~0.70   |
| Random Forest      | ~0.75   |
| XGBoost            | ~0.77   |

**Feature Importances (CCHS & BRFSS Comparison)**  
Top predictors across both datasets:  
- BMI (measured or self-reported)  
- High blood pressure  
- High cholesterol  

Note: Differences exist in smoking and physical activity due to dataset definitions.

## Model Interpretability  

**Random Forest and XGBoost**  
- Feature importance plots highlight BMI, high blood pressure, and cholesterol  
- SHAP values show feature-level impacts on individual predictions  
- Confirms BMI and hypertension as strong diabetes predictors  

## Key Takeaways  

- Consistent risk factors: BMI, high blood pressure, and cholesterol are strongly linked to diabetes in both datasets  
- Variable definitions: CCHS offers richer categorical detail; BRFSS uses simpler binary coding  
- Data harmonization challenge: Cross-country comparison requires careful handling of survey methodology differences  
- Model performance: XGBoost consistently outperforms Logistic Regression and Random Forest  

---


