# 🩺 Diabetes Risk Prediction Dashboard

An interactive web dashboard for analyzing diabetes risk factors using the BRFSS (Behavioral Risk Factor Surveillance System) dataset from the United States.

## 🌐 **Live Dashboard Access**

**🔗 [View Dashboard on GitHub](https://github.com/DanielDemoz/Diabetes-Risk-Prediction-Using-BRFSS-US-and-CCHS-Canada)**

**📱 [Open Static Dashboard](https://github.com/DanielDemoz/Diabetes-Risk-Prediction-Using-BRFSS-US-and-CCHS-Canada/blob/main/static_dashboard.html)** - Click "Download" then open the HTML file in your browser for immediate access!

## 🚀 **Quick Start (No Installation Required)**

1. **Download** the `static_dashboard.html` file from the repository
2. **Double-click** the file to open it in your browser
3. **Start exploring** the interactive dashboard immediately!

## 🚀 Features

- **📊 Overview Page**: Key statistics and dataset information
- **🔍 Exploratory Data Analysis**: Interactive visualizations with filtering options
- **🤖 Model Comparison**: Performance metrics for multiple ML models
- **🔮 Risk Prediction**: Personalized diabetes risk assessment

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Dashboard

1. **Ensure your data file is in the correct location:**
   - Place `diabetes_012_health_indicators_BRFSS2015.csv` in the `data/` folder
   - The file should be located at: `data/diabetes_012_health_indicators_BRFSS2015.csv`

2. **Run the Streamlit application:**
   ```bash
   streamlit run dashboard.py
   ```

3. **Open your browser:**
   - The dashboard will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

## 📁 Project Structure

```
├── dashboard.py                              # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv  # Dataset file
├── DASHBOARD_README.md                       # This file
└── README.md                                 # Original project README
```

## 🎯 Dashboard Pages

### 📊 Overview
- Dataset summary statistics
- Key health metrics
- Diabetes status distribution
- Feature correlation analysis

### 🔍 Exploratory Data Analysis
- Interactive filtering by age, BMI, and sex
- Distribution plots for BMI and age
- Risk factor prevalence analysis
- Lifestyle factors comparison
- Correlation heatmap

### 🤖 Model Comparison
- Performance metrics for three ML models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Feature importance analysis
- Confusion matrices
- ROC-AUC comparison

### 🔮 Risk Prediction
- Interactive form for health information input
- Personalized risk assessment
- Risk factor identification
- Health recommendations

## 🔧 Technical Details

### Models Used
- **Logistic Regression**: Baseline interpretable model
- **Random Forest**: Ensemble method for non-linear relationships
- **XGBoost**: Gradient boosting for optimal performance

### Data Preprocessing
- Class imbalance handling with SMOTE
- Feature scaling and encoding
- Train-test split (70-30)

### Key Features
- BMI (Body Mass Index)
- High blood pressure
- High cholesterol
- Smoking status
- Physical activity
- Fruit and vegetable consumption
- Age, sex, education, income

## 📊 Dataset Information

**Source**: BRFSS (Behavioral Risk Factor Surveillance System) 2015
**Size**: ~253,680 records
**Target Variable**: Diabetes status (0: No diabetes, 1: Prediabetes, 2: Diabetes)

## 🎨 Features

- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Visualizations**: Plotly charts with zoom, pan, and hover
- **Real-time Filtering**: Dynamic data filtering in EDA page
- **Caching**: Optimized performance with Streamlit caching
- **Modern UI**: Clean, professional interface with custom CSS

## 🔍 Usage Tips

1. **Navigation**: Use the sidebar to switch between pages
2. **Filtering**: In the EDA page, use sidebar filters to explore specific populations
3. **Predictions**: Fill out all fields in the prediction page for accurate results
4. **Interpretation**: Risk levels are color-coded (Green: Low, Yellow: Moderate, Red: High)

## 🚨 Important Notes

- This dashboard is for educational and research purposes
- Medical decisions should not be based solely on this tool
- Consult healthcare professionals for medical advice
- The models are trained on 2015 data and may not reflect current health trends

## 🐛 Troubleshooting

**Common Issues:**

1. **File not found error**: Ensure the CSV file is in the `data/` folder
2. **Port already in use**: Try `streamlit run dashboard.py --server.port 8502`
3. **Memory issues**: The dataset is large; ensure sufficient RAM (4GB+ recommended)

## 📈 Performance

- **Loading time**: ~10-15 seconds for initial data load
- **Model training**: ~30-60 seconds (cached after first run)
- **Prediction**: Near-instantaneous

## 🔮 Future Enhancements

- Integration with CCHS (Canadian) dataset
- Additional ML models (SVM, Neural Networks)
- Export functionality for reports
- User authentication and data persistence
- API endpoints for external integration

---

**Built with ❤️ using Streamlit, Plotly, and Scikit-learn**
