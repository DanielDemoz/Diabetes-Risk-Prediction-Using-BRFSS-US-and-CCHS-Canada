# 🩺 Diabetes Risk Prediction Dashboard
**Author:** Daniel S. Demoz

## 🚀 **LIVE INTERACTIVE DASHBOARD**

**🌐 [CLICK HERE TO ACCESS THE DASHBOARD](https://danieldemoz.github.io/Diabetes-Risk-Prediction-Using-BRFSS-US-and-CCHS-Canada/)**

*No downloads required - works instantly in any web browser!*

## 📊 **Dashboard Features**

### **🎯 Interactive Pages:**
- **📈 Overview**: Dataset statistics and diabetes distribution
- **🔍 EDA**: Interactive data exploration with filtering
- **🤖 Models**: Machine learning model performance comparison
- **🔮 Prediction**: Personalized diabetes risk assessment

### **💡 Key Features:**
- **Real-time visualizations** with Plotly charts
- **Interactive filtering** and data exploration
- **Personalized risk prediction** based on health factors
- **Professional design** with responsive layout
- **No installation required** - works in any browser

## 📋 **About the Project**

This comprehensive dashboard analyzes diabetes risk factors using **two major health survey datasets** from North America, enabling cross-country comparison and validation of diabetes prediction models.

### **🌍 Why Both US and Canadian Data?**

**Cross-Country Validation**: By using datasets from both the United States and Canada, this project provides:
- **Validation of risk factors** across different healthcare systems
- **Comparison of model performance** in different populations
- **Insights into cultural and systemic differences** in health data collection
- **Robustness testing** of machine learning models across borders

### **📊 Dataset Sources & Integration**

#### **🇺🇸 US Dataset: BRFSS (Behavioral Risk Factor Surveillance System) 2015**
- **Source**: [CDC BRFSS](https://www.cdc.gov/brfss/)
- **Records**: 253,680 individuals
- **Coverage**: All 50 US states, DC, and territories
- **Features**: BMI, blood pressure, cholesterol, smoking, physical activity, diet, demographics

#### **🇨🇦 Canadian Dataset: CCHS (Canadian Community Health Survey)**
- **Source**: [Statistics Canada](https://www150.statcan.gc.ca/n1/pub/82m0013x/82m0013x2024001-eng.htm)
- **Records**: 108,252 individuals
- **Coverage**: All Canadian provinces and territories
- **Features**: Self-reported and adjusted BMI, detailed smoking status, WHO physical activity guidelines, fruit/vegetable consumption

### **🔗 Data Integration Strategy**

**Harmonization Process**:
1. **Column Mapping**: Aligned similar health indicators across datasets
2. **Value Recoding**: Standardized categorical variables (diabetes, high BP, cholesterol)
3. **Missing Data Handling**: Different strategies for each dataset's unique coding
4. **Feature Engineering**: Created comparable variables for cross-dataset analysis

**Key Differences Handled**:
- **BMI**: BRFSS (measured) vs CCHS (self-reported + adjusted)
- **Smoking**: BRFSS (binary) vs CCHS (detailed categories)
- **Physical Activity**: BRFSS (binary) vs CCHS (minutes + WHO guidelines)

## 📈 **Cross-Country Model Performance Comparison**

### **🎯 Model Performance Results (ROC-AUC Scores)**

| Model | 🇺🇸 BRFSS (US) | 🇨🇦 CCHS (Canada) | Performance Gap |
|-------|----------------|-------------------|-----------------|
| **Logistic Regression** | 0.8149 | 0.7729 | +0.0420 |
| **Random Forest** | 0.7913 | 0.7055 | +0.0858 |
| **XGBoost** | 0.8206 | 0.7315 | +0.0891 |

### **🔍 Key Findings**

**Consistent Risk Factors Across Countries**:
- **High Blood Pressure**: Most important predictor in both datasets
- **BMI**: Strong predictor in both US (measured) and Canada (self-reported)
- **High Cholesterol**: Significant risk factor in both populations

**Model Performance Insights**:
- **Logistic Regression**: Most consistent performance across countries
- **Tree-based Models**: Better performance on US data, likely due to data granularity differences
- **Overall**: Models show comparable predictive ability with slight US advantage

### **📊 Dataset Comparison**

| Aspect | 🇺🇸 BRFSS (US) | 🇨🇦 CCHS (Canada) |
|--------|----------------|-------------------|
| **Sample Size** | 253,680 | 108,252 |
| **Data Collection** | Phone surveys | Mixed methods |
| **BMI Measurement** | Measured/calculated | Self-reported + adjusted |
| **Smoking Detail** | Binary (yes/no) | Detailed categories |
| **Physical Activity** | Binary engagement | Minutes + WHO guidelines |
| **Missing Data** | Minimal | Coded as special values |

## 🛠️ **Technical Details**

**Built with:**
- HTML5, CSS3, JavaScript
- Plotly.js for interactive visualizations
- Responsive design for all devices

**Data Sources:**
- **US**: BRFSS 2015 (253,680+ records)
- **Canada**: CCHS (108,252+ records)

**Models Used:**
- Logistic Regression, Random Forest, XGBoost
- Cross-validated on both datasets
- SMOTE for class imbalance handling

## 🔬 **Research Significance**

### **Why This Cross-Country Analysis Matters**

**Healthcare System Comparison**:
- **US**: Market-based healthcare system
- **Canada**: Universal healthcare system
- **Impact**: Different access patterns may affect health outcomes and data quality

**Methodological Validation**:
- **Cross-validation**: Models trained on one country tested on another
- **Feature consistency**: Same risk factors identified across different populations
- **Robustness**: Models perform well despite data collection differences

**Public Health Insights**:
- **Universal risk factors**: BMI, blood pressure, cholesterol are consistent predictors
- **Cultural differences**: Smoking and physical activity patterns vary between countries
- **Data quality**: Different survey methodologies provide complementary insights

### **📚 Data Sources & References**

**Primary Sources**:
1. **BRFSS 2015**: [CDC Behavioral Risk Factor Surveillance System](https://www.cdc.gov/brfss/)
2. **CCHS**: [Statistics Canada - Canadian Community Health Survey](https://www150.statcan.gc.ca/n1/pub/82m0013x/82m0013x2024001-eng.htm)

**Additional References**:
- [Health Data Nexus - CCHS Documentation](https://healthdatanexus.ai/content/cchs-pumf/1.1/)
- [Kaggle - Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

## 📞 **Contact**

**Author:** Daniel S. Demoz  
**Repository:** [GitHub](https://github.com/DanielDemoz/Diabetes-Risk-Prediction-Using-BRFSS-US-and-CCHS-Canada)

---

*This dashboard is for educational and research purposes. Medical decisions should not be based solely on this tool. Please consult healthcare professionals for medical advice.*


