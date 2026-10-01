import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff

st.set_page_config(page_title="AESIF | Risk Scoring", layout="wide")
st.title("⚠️ Project Risk Assessment (XGBoost)")

st.markdown("""
Upload historical project data to train the XGBoost classifier. The model identifies complex nonlinear 
relationships to predict **Overrun Risk** (0 = On-Target, 1 = High Risk of Overrun).
""")

uploaded_file = st.file_uploader("Upload Historical Risk Data (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head(5))
    
    target_col = st.selectbox("Select Target Variable (Risk Label)", df.columns)
    feature_cols = st.multiselect("Select Feature Columns (e.g., CAPEX, Duration, Complexity)", df.columns, default=[c for c in df.columns if c != target_col])
    
    if st.button("Initialize & Train XGBoost Model"):
        with st.spinner("Training Extreme Gradient Boosting Model..."):
            X = df[feature_cols]
            y = df[target_col]
            
            # Handle categorical encoding for the fly demo
            X = pd.get_dummies(X, drop_first=True)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # XGBoost Model configuration
            model = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=5, 
                learning_rate=0.1, 
                objective='binary:logistic'
            )
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            
            st.success(f"Model trained successfully! Test Accuracy: {accuracy:.2%}")
            
            # Feature Importance Chart
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Drivers of Project Risk")
            st.plotly_chart(fig, use_container_width=True)

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)
            fig_cm = ff.create_annotated_heatmap(z=cm, x=['Pred: Safe', 'Pred: Risk'], y=['True: Safe', 'True: Risk'], colorscale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
