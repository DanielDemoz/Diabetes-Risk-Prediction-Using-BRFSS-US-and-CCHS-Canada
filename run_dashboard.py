#!/usr/bin/env python3
"""
Simple launcher script for the Diabetes Risk Prediction Dashboard
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import plotly
        import sklearn
        import xgboost
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def check_data_file():
    """Check if the data file exists"""
    data_path = "data/diabetes_012_health_indicators_BRFSS2015.csv"
    if os.path.exists(data_path):
        print("✅ Data file found")
        return True
    else:
        print(f"❌ Data file not found at: {data_path}")
        print("Please ensure the CSV file is in the data/ folder")
        return False

def main():
    """Main launcher function"""
    print("🩺 Diabetes Risk Prediction Dashboard Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check data file
    if not check_data_file():
        sys.exit(1)
    
    print("\n🚀 Starting dashboard...")
    print("The dashboard will open in your default browser")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
