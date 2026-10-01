import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AESIF | Analytics", layout="wide")
st.title("📊 Dynamic Project Analytics")

uploaded_file = st.file_uploader("Upload Project Dataset (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("Dataset loaded successfully.")
        
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head(10))
            
        st.subheader("On-the-Fly Visualization")
        col1, col2, col3 = st.columns(3)
        
        # Dynamic UI controls based on uploaded dataset columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        with col1:
            chart_type = st.selectbox("Chart Type", ["Scatter Plot", "Bar Chart", "Line Chart", "Box Plot"])
        with col2:
            x_axis = st.selectbox("X-Axis", df.columns.tolist())
        with col3:
            y_axis = st.selectbox("Y-Axis", numeric_cols)
            
        color_opt = st.selectbox("Color By (Optional)", ["None"] + categorical_cols)
        color_col = None if color_opt == "None" else color_opt

        # Generate Plotly charts dynamically
        if chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, template="plotly_white")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, template="plotly_white")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, color=color_col, template="plotly_white")
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis, color=color_col, template="plotly_white")
            
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a dataset to begin interactive analysis.")
