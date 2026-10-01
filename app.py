import streamlit as st

# Configure global page settings
st.set_page_config(
    page_title="AESIF | Capital Project Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("⚡ AI-Enhanced Strategic Investment Framework (AESIF)")
    st.markdown("### Executive Dashboard for Capital Energy Transition Projects")
    
    st.markdown("""
    Welcome to the AESIF Live Environment. This framework leverages advanced machine learning 
    to drive decision-making in capital-intensive projects. 
    
    **Capabilities within this environment:**
    * **Exploratory Analytics:** Upload custom datasets (CSV/Excel) to generate on-the-fly interactive charts.
    * **XGBoost Risk Scoring:** Evaluate project viability and predict schedule/cost overrun probabilities using gradient boosting.
    * **LSTM Cost Forecasting:** Utilize deep learning (Long Short-Term Memory networks) for time-series forecasting of capital expenditures and resource curves.
    
    👈 **Select a module from the sidebar to begin analysis.**
    """)
    
    st.info("System Status: All AI Agents Online. Ready for live dataset ingestion.")

    # High-level placeholder metrics for the landing page
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Active Portfolios", value="4", delta="1 New")
    col2.metric(label="Total CAPEX Tracked", value="$1.2B", delta="3.2%")
    col3.metric(label="Risk Alerts", value="2", delta="-1", delta_color="inverse")
    col4.metric(label="Model Accuracy (Avg)", value="94.2%", delta="1.1%")

if __name__ == "__main__":
    main()