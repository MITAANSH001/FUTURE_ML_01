import streamlit as st
import os
import pandas as pd
from src.data_processor import DataProcessor
from src.model import SalesForecaster
from src.utils import Visualizer

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# 1. Sidebar for User Interaction
st.sidebar.header("🛠️ Dashboard Settings")
DATA_PATH = "data/sample_sales.csv"

# Slider to let user choose how many days to forecast
forecast_days = st.sidebar.slider("Select Forecast Horizon (Days)", 7, 30, 14)

if os.path.exists(DATA_PATH):
    processor = DataProcessor(DATA_PATH)
    forecaster = SalesForecaster()
    visualizer = Visualizer()

    df = processor.load_data()
    ts_data = processor.get_time_series_data()

    st.title("📊 Sales & Demand Forecasting Dashboard")
    st.markdown(f"**Current Dataset:** {DATA_PATH} ({len(ts_data)} days of history)")

    # 2. Dynamic Model Training
    metrics = forecaster.train_arima(ts_data)
    
    # Professional Metric Cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy (MAPE)", f"{metrics['mape']:.2f}%", help="Lower is better")
    c2.metric("RMSE Error", f"{metrics['rmse']:.2f}")
    c3.metric("Model Status", metrics['status'])

    # 3. Interactive Plotting
    st.subheader(f"📈 {forecast_days}-Day AI Demand Forecast")
    future = forecaster.predict_future(ts_data, steps=forecast_days)
    fig = visualizer.plot_future_forecast(ts_data, future)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Data Export Feature
    st.divider()
    st.subheader("📥 Export Forecast Results")
    forecast_df = pd.DataFrame({'Date': future.index, 'Predicted_Sales': future.values})
    
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name='future_sales_forecast.csv',
        mime='text/csv',
    )
else:
    st.error(f"File not found at {DATA_PATH}.")
