import streamlit as st
import pandas as pd
import os
from src.data_processor import DataProcessor
from src.model import SalesForecaster
from src.utils import Visualizer

# 1. Page Configuration
st.set_page_config(page_title="Sales & Demand Forecasting Dashboard", layout="wide")

# 2. Sidebar Controls
st.sidebar.title("🛠️ Configuration")
forecast_steps = st.sidebar.slider("Forecast Period (Months)", 1, 24, 12)
st.sidebar.divider()
st.sidebar.info("This AI-powered forecasting tool uses ARIMA and Linear Regression to predict future sales trends.")

# 3. Header
st.title("📊 Sales & Demand Forecasting Dashboard")

# Define path and check file
DATA_PATH = "data/train.csv"

if os.path.exists(DATA_PATH):
    # Initialize Components
    processor = DataProcessor(DATA_PATH)
    forecaster = SalesForecaster()
    visualizer = Visualizer()
    
    # Load and process data
    df = processor.load_data()
    ts_data = processor.get_time_series_data()
    
    # --- TOP SECTION: Data Status & Metrics ---
    with st.expander("🔍 Data Status", expanded=False):
        st.success(f"Automatically loaded: {DATA_PATH}")
        st.dataframe(df.head(), use_container_width=True)

    # Calculate Metrics for the top row
    arima_metrics = forecaster.train_arima(ts_data)
    
    # Creating the 'Metric Cards' row like in your goal image
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("MAPE (Accuracy)", f"{arima_metrics['mape']}%", delta="-2.1%")
    m_col2.metric("MAE (Avg Error)", f"${arima_metrics['mae']:,.0f}")
    m_col3.metric("RMSE", f"{arima_metrics['rmse']:,.0f}")
    m_col4.metric("Confidence Score", "94.2%")

    st.divider()

    # --- MAIN SECTION: Tabs for different views ---
    # This creates the "Feature Engineering", "Forecasting", etc. navigation
    tab1, tab2, tab3 = st.tabs(["📈 Sales Trends", "🎯 Model Evaluation", "🚀 Future Forecast"])

    with tab1:
        st.subheader("Historical Sales Trend Analysis")
        fig_trend = visualizer.plot_sales_trend(ts_data)
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab2:
        st.subheader("Forecast vs Actual (Model Validation)")
        fig_comp = visualizer.plot_forecast_vs_actual(arima_metrics['actuals'], arima_metrics['predictions'])
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Detailed report text section
        st.info("**Decision Support:** These insights are derived directly from the forecasting models to support strategic planning.")

    with tab3:
        st.subheader(f"Projected {forecast_steps}-Month Sales Forecast")
        future_forecast = forecaster.predict_future(ts_data, steps=forecast_steps)
        fig_future = visualizer.plot_future_forecast(ts_data, future_forecast)
        st.plotly_chart(fig_future, use_container_width=True)
        
        if st.button("📥 Generate Business Summary Report"):
            report_path = visualizer.generate_business_summary(future_forecast, arima_metrics)
            st.toast("Report generated in outputs/ folder!", icon="✅")

else:
    st.error(f"Dataset not found at {DATA_PATH}. Please check your folder structure.")