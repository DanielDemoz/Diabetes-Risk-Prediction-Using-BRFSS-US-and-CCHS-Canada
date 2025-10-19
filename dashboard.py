import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the BRFSS dataset"""
    try:
        # Load the dataset
        df = pd.read_csv('data/diabetes_012_health_indicators_BRFSS2015.csv')
        
        # Rename columns for clarity
        df = df.rename(columns={
            'Diabetes_012': 'diabetes',
            'HighBP': 'high_bp',
            'HighChol': 'high_chol',
            'BMI': 'bmi',
            'Smoker': 'smoker',
            'PhysActivity': 'phys_activity',
            'Fruits': 'fruits',
            'Veggies': 'veggies'
        })
        
        # Convert diabetes to binary (0: No diabetes, 1: Prediabetes/Diabetes)
        df['diabetes_binary'] = df['diabetes'].replace({1.0: 1, 2.0: 1, 0.0: 0})
        
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'diabetes_012_health_indicators_BRFSS2015.csv' is in the data folder.")
        return None

@st.cache_data
def get_data_summary(df):
    """Get summary statistics for the dataset"""
    summary = {
        'total_records': len(df),
        'diabetes_cases': df['diabetes_binary'].sum(),
        'diabetes_rate': df['diabetes_binary'].mean() * 100,
        'avg_bmi': df['bmi'].mean(),
        'high_bp_rate': df['high_bp'].mean() * 100,
        'high_chol_rate': df['high_chol'].mean() * 100,
        'smoker_rate': df['smoker'].mean() * 100,
        'phys_activity_rate': df['phys_activity'].mean() * 100
    }
    return summary

def create_overview_page(df, summary):
    """Create the overview page"""
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 Dataset Overview
    This dashboard analyzes diabetes risk factors using the BRFSS (Behavioral Risk Factor Surveillance System) dataset from the United States.
    The dataset contains health indicators for over 250,000 individuals and includes various risk factors for diabetes.
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Total Records",
            value=f"{summary['total_records']:,}",
            help="Total number of individuals in the dataset"
        )
    
    with col2:
        st.metric(
            label="🩸 Diabetes Cases",
            value=f"{summary['diabetes_cases']:,}",
            delta=f"{summary['diabetes_rate']:.1f}%",
            help="Number and percentage of individuals with diabetes or prediabetes"
        )
    
    with col3:
        st.metric(
            label="⚖️ Average BMI",
            value=f"{summary['avg_bmi']:.1f}",
            help="Average Body Mass Index across all individuals"
        )
    
    with col4:
        st.metric(
            label="💓 High BP Rate",
            value=f"{summary['high_bp_rate']:.1f}%",
            help="Percentage of individuals with high blood pressure"
        )
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="🧬 High Cholesterol",
            value=f"{summary['high_chol_rate']:.1f}%",
            help="Percentage of individuals with high cholesterol"
        )
    
    with col6:
        st.metric(
            label="🚬 Smoker Rate",
            value=f"{summary['smoker_rate']:.1f}%",
            help="Percentage of current smokers"
        )
    
    with col7:
        st.metric(
            label="🏃 Physical Activity",
            value=f"{summary['phys_activity_rate']:.1f}%",
            help="Percentage of individuals who engage in physical activity"
        )
    
    with col8:
        st.metric(
            label="🥗 Fruit/Vegetable Intake",
            value=f"{(df['fruits'].mean() + df['veggies'].mean())/2:.1f}",
            help="Average fruit and vegetable consumption score"
        )
    
    # Dataset information
    st.markdown("## 📋 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Target Variable")
        st.markdown("""
        - **0**: No diabetes
        - **1**: Prediabetes  
        - **2**: Diabetes
        """)
        
        # Diabetes distribution
        diabetes_counts = df['diabetes'].value_counts().sort_index()
        fig = px.pie(
            values=diabetes_counts.values,
            names=['No Diabetes', 'Prediabetes', 'Diabetes'],
            title="Diabetes Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Key Features")
        st.markdown("""
        - **BMI**: Body Mass Index
        - **High BP**: High blood pressure (binary)
        - **High Chol**: High cholesterol (binary)
        - **Smoker**: Current smoking status (binary)
        - **Phys Activity**: Physical activity engagement (binary)
        - **Fruits/Veggies**: Daily fruit and vegetable consumption
        - **Age, Sex, Education, Income**: Demographic variables
        """)
        
        # Feature correlation with diabetes
        numeric_cols = ['bmi', 'high_bp', 'high_chol', 'smoker', 'phys_activity', 'fruits', 'veggies']
        correlations = df[numeric_cols + ['diabetes_binary']].corr()['diabetes_binary'].drop('diabetes_binary').sort_values(key=abs, ascending=True)
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title="Feature Correlation with Diabetes",
            color=correlations.values,
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_eda_page(df):
    """Create the exploratory data analysis page"""
    st.markdown("# 🔍 Exploratory Data Analysis")
    
    # Sidebar filters
    st.sidebar.markdown("## 🔧 Filters")
    
    # Age filter
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )
    
    # BMI filter
    bmi_range = st.sidebar.slider(
        "BMI Range",
        min_value=float(df['bmi'].min()),
        max_value=float(df['bmi'].max()),
        value=(float(df['bmi'].min()), float(df['bmi'].max()))
    )
    
    # Sex filter
    sex_filter = st.sidebar.selectbox(
        "Sex",
        options=['All', 'Male', 'Female'],
        index=0
    )
    
    # Apply filters
    filtered_df = df[
        (df['Age'] >= age_range[0]) & 
        (df['Age'] <= age_range[1]) &
        (df['bmi'] >= bmi_range[0]) & 
        (df['bmi'] <= bmi_range[1])
    ]
    
    if sex_filter != 'All':
        sex_value = 1 if sex_filter == 'Male' else 0
        filtered_df = filtered_df[filtered_df['Sex'] == sex_value]
    
    st.markdown(f"**Filtered Dataset**: {len(filtered_df):,} records")
    
    # Distribution plots
    st.markdown("## 📈 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # BMI distribution by diabetes status
        fig = px.histogram(
            filtered_df,
            x='bmi',
            color='diabetes',
            nbins=30,
            title="BMI Distribution by Diabetes Status",
            labels={'diabetes': 'Diabetes Status'},
            color_discrete_map={0: 'lightblue', 1: 'orange', 2: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution by diabetes status
        fig = px.histogram(
            filtered_df,
            x='Age',
            color='diabetes',
            nbins=20,
            title="Age Distribution by Diabetes Status",
            labels={'diabetes': 'Diabetes Status'},
            color_discrete_map={0: 'lightblue', 1: 'orange', 2: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factor analysis
    st.markdown("## ⚠️ Risk Factor Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk factors by diabetes status
        risk_factors = ['high_bp', 'high_chol', 'smoker']
        risk_data = []
        
        for factor in risk_factors:
            for diabetes_status in [0, 1, 2]:
                rate = filtered_df[filtered_df['diabetes'] == diabetes_status][factor].mean() * 100
                risk_data.append({
                    'Risk Factor': factor.replace('_', ' ').title(),
                    'Diabetes Status': ['No Diabetes', 'Prediabetes', 'Diabetes'][diabetes_status],
                    'Rate (%)': rate
                })
        
        risk_df = pd.DataFrame(risk_data)
        
        fig = px.bar(
            risk_df,
            x='Risk Factor',
            y='Rate (%)',
            color='Diabetes Status',
            title="Risk Factor Prevalence by Diabetes Status",
            color_discrete_map={'No Diabetes': 'lightblue', 'Prediabetes': 'orange', 'Diabetes': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Physical activity and diet
        lifestyle_factors = ['phys_activity', 'fruits', 'veggies']
        lifestyle_data = []
        
        for factor in lifestyle_factors:
            for diabetes_status in [0, 1, 2]:
                if factor == 'phys_activity':
                    rate = filtered_df[filtered_df['diabetes'] == diabetes_status][factor].mean() * 100
                else:
                    rate = filtered_df[filtered_df['diabetes'] == diabetes_status][factor].mean()
                lifestyle_data.append({
                    'Lifestyle Factor': factor.replace('_', ' ').title(),
                    'Diabetes Status': ['No Diabetes', 'Prediabetes', 'Diabetes'][diabetes_status],
                    'Average': rate
                })
        
        lifestyle_df = pd.DataFrame(lifestyle_data)
        
        fig = px.bar(
            lifestyle_df,
            x='Lifestyle Factor',
            y='Average',
            color='Diabetes Status',
            title="Lifestyle Factors by Diabetes Status",
            color_discrete_map={'No Diabetes': 'lightblue', 'Prediabetes': 'orange', 'Diabetes': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("## 🔗 Correlation Analysis")
    
    numeric_cols = ['bmi', 'high_bp', 'high_chol', 'smoker', 'phys_activity', 'fruits', 'veggies', 'Age', 'diabetes_binary']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def train_models(df):
    """Train machine learning models"""
    # Prepare features and target
    feature_cols = ['high_bp', 'high_chol', 'bmi', 'smoker', 'phys_activity', 'fruits', 'veggies', 'Age', 'Sex', 'Education', 'Income']
    X = df[feature_cols]
    y = df['diabetes_binary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Train models
    models = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_res, y_train_res)
    models['Logistic Regression'] = lr
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)
    models['Random Forest'] = rf
    
    # XGBoost
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb.fit(X_train_res, y_train_res)
    models['XGBoost'] = xgb
    
    # Get predictions
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        predictions[name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
    
    return models, predictions, feature_cols

def create_model_page(df):
    """Create the model comparison page"""
    st.markdown("# 🤖 Model Performance Comparison")
    
    # Train models
    with st.spinner("Training models... This may take a moment."):
        models, predictions, feature_cols = train_models(df)
    
    # Model performance metrics
    st.markdown("## 📊 Model Performance Metrics")
    
    performance_data = []
    for name, pred_data in predictions.items():
        y_test = pred_data['y_test']
        y_pred = pred_data['y_pred']
        y_pred_proba = pred_data['y_pred_proba']
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        performance_data.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    for i, (_, row) in enumerate(performance_df.iterrows()):
        with [col1, col2, col3][i]:
            st.metric(
                label=f"**{row['Model']}**",
                value=f"{row['ROC AUC']:.3f}",
                delta=f"F1: {row['F1 Score']:.3f}",
                help=f"Accuracy: {row['Accuracy']:.3f}"
            )
    
    # Performance comparison chart
    fig = px.bar(
        performance_df,
        x='Model',
        y=['Accuracy', 'F1 Score', 'ROC AUC'],
        title="Model Performance Comparison",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("## 🎯 Feature Importance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Random Forest feature importance
        rf_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': models['Random Forest'].feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            rf_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Random Forest Feature Importance",
            color='Importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': models['XGBoost'].feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            xgb_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="XGBoost Feature Importance",
            color='Importance',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.markdown("## 🔍 Confusion Matrices")
    
    cols = st.columns(len(models))
    for i, (name, pred_data) in enumerate(predictions.items()):
        with cols[i]:
            cm = confusion_matrix(pred_data['y_test'], pred_data['y_pred'])
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"{name} Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

def create_prediction_page(df):
    """Create the prediction page"""
    st.markdown("# 🔮 Diabetes Risk Prediction")
    
    st.markdown("""
    Enter your health information below to get a personalized diabetes risk prediction.
    The model will analyze your risk factors and provide a probability score.
    """)
    
    # Train models for prediction
    with st.spinner("Loading prediction models..."):
        models, _, feature_cols = train_models(df)
    
    # Input form
    st.markdown("## 📝 Health Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏥 Medical History")
        high_bp = st.selectbox("High Blood Pressure", ["No", "Yes"])
        high_chol = st.selectbox("High Cholesterol", ["No", "Yes"])
        bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        
        st.markdown("### 🚬 Lifestyle Factors")
        smoker = st.selectbox("Current Smoker", ["No", "Yes"])
        phys_activity = st.selectbox("Physical Activity", ["No", "Yes"])
    
    with col2:
        st.markdown("### 🥗 Diet")
        fruits = st.slider("Daily Fruit Consumption (0-30)", min_value=0, max_value=30, value=1)
        veggies = st.slider("Daily Vegetable Consumption (0-30)", min_value=0, max_value=30, value=1)
        
        st.markdown("### 👤 Demographics")
        age = st.slider("Age", min_value=18, max_value=100, value=45)
        sex = st.selectbox("Sex", ["Female", "Male"])
        education = st.selectbox("Education Level", 
                               ["Never attended school", "Elementary", "Some high school", 
                                "High school graduate", "Some college", "College graduate"])
        income = st.selectbox("Income Level", 
                            ["Less than $10,000", "$10,000-$15,000", "$15,000-$20,000",
                             "$20,000-$25,000", "$25,000-$35,000", "$35,000-$50,000",
                             "$50,000-$75,000", "$75,000 or more"])
    
    # Convert inputs to model format
    input_data = {
        'high_bp': 1 if high_bp == "Yes" else 0,
        'high_chol': 1 if high_chol == "Yes" else 0,
        'bmi': bmi,
        'smoker': 1 if smoker == "Yes" else 0,
        'phys_activity': 1 if phys_activity == "Yes" else 0,
        'fruits': fruits,
        'veggies': veggies,
        'Age': age,
        'Sex': 1 if sex == "Male" else 0,
        'Education': ["Never attended school", "Elementary", "Some high school", 
                     "High school graduate", "Some college", "College graduate"].index(education),
        'Income': ["Less than $10,000", "$10,000-$15,000", "$15,000-$20,000",
                  "$20,000-$25,000", "$25,000-$35,000", "$35,000-$50,000",
                  "$50,000-$75,000", "$75,000 or more"].index(income)
    }
    
    # Create prediction button
    if st.button("🔮 Predict Diabetes Risk", type="primary"):
        # Prepare input for prediction
        input_df = pd.DataFrame([input_data])
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            prob = model.predict_proba(input_df)[0, 1]
            predictions[name] = prob
        
        # Display results
        st.markdown("## 📊 Prediction Results")
        
        # Risk level interpretation
        avg_prob = np.mean(list(predictions.values()))
        
        if avg_prob < 0.3:
            risk_level = "🟢 Low Risk"
            risk_color = "green"
        elif avg_prob < 0.6:
            risk_level = "🟡 Moderate Risk"
            risk_color = "orange"
        else:
            risk_level = "🔴 High Risk"
            risk_color = "red"
        
        # Display risk level
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 1rem; margin: 1rem 0;">
            <h2 style="color: {risk_color};">{risk_level}</h2>
            <h3>Average Risk Probability: {avg_prob:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual model predictions
        st.markdown("### 🤖 Model Predictions")
        
        cols = st.columns(len(models))
        for i, (name, prob) in enumerate(predictions.items()):
            with cols[i]:
                st.metric(
                    label=name,
                    value=f"{prob:.1%}",
                    help=f"Probability of diabetes risk"
                )
        
        # Risk factors analysis
        st.markdown("### ⚠️ Risk Factor Analysis")
        
        risk_factors = []
        if input_data['high_bp']:
            risk_factors.append("High Blood Pressure")
        if input_data['high_chol']:
            risk_factors.append("High Cholesterol")
        if input_data['bmi'] > 30:
            risk_factors.append("Obesity (BMI > 30)")
        elif input_data['bmi'] > 25:
            risk_factors.append("Overweight (BMI > 25)")
        if input_data['smoker']:
            risk_factors.append("Smoking")
        if not input_data['phys_activity']:
            risk_factors.append("Physical Inactivity")
        if input_data['fruits'] < 2:
            risk_factors.append("Low Fruit Consumption")
        if input_data['veggies'] < 2:
            risk_factors.append("Low Vegetable Consumption")
        if input_data['Age'] > 65:
            risk_factors.append("Advanced Age (>65)")
        
        if risk_factors:
            st.markdown("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("✅ No major risk factors identified")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recommendations = []
        if input_data['bmi'] > 25:
            recommendations.append("Consider weight management and healthy eating")
        if not input_data['phys_activity']:
            recommendations.append("Increase physical activity (aim for 150 minutes/week)")
        if input_data['fruits'] < 2 or input_data['veggies'] < 2:
            recommendations.append("Increase fruit and vegetable consumption")
        if input_data['smoker']:
            recommendations.append("Consider smoking cessation programs")
        if input_data['high_bp'] or input_data['high_chol']:
            recommendations.append("Consult with healthcare provider about cardiovascular health")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.markdown("✅ Continue maintaining your healthy lifestyle!")

def main():
    """Main application function"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Overview", "🔍 EDA", "🤖 Models", "🔮 Prediction"]
    )
    
    # Get data summary
    summary = get_data_summary(df)
    
    # Route to appropriate page
    if page == "📊 Overview":
        create_overview_page(df, summary)
    elif page == "🔍 EDA":
        create_eda_page(df)
    elif page == "🤖 Models":
        create_model_page(df)
    elif page == "🔮 Prediction":
        create_prediction_page(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **🩺 Diabetes Risk Prediction Dashboard**
    
    Built with Streamlit for analyzing BRFSS health data.
    
    **Data Source:** BRFSS 2015
    **Models:** Logistic Regression, Random Forest, XGBoost
    """)

if __name__ == "__main__":
    main()
