# 📊 FUTURE_ML_01: Sales & Demand Forecasting Dashboard

## 🚀 Project Overview
This repository contains the **Sales & Demand Forecasting Dashboard** developed as part of my AI/ML Internship at **Future Interns**. The project is an interactive decision-support tool designed to predict future sales trends using historical data and advanced machine learning models.

## 🛠️ Key Features
- **Automated Data Processing**: Performs data cleaning, missing value imputation, and time-series resampling automatically upon file detection.
- **AI-Powered Forecasting**: Implements **ARIMA** (AutoRegressive Integrated Moving Average) and **Linear Regression** for precise time-series predictions.
- **Interactive Dashboard**: A sleek, dark-themed UI built with **Streamlit** featuring dynamic **Plotly** visualizations.
- **Business Intelligence**: Automatically generates **MAPE**, **RMSE**, and **MAE** metrics to validate model accuracy.
- **Automated Reporting**: Generates a business-ready Markdown summary with strategic inventory recommendations.

## 📂 Project Structure
- `app.py`: The main entry point for the Streamlit dashboard application.
- `src/`: Modular Python logic for scalability:
    - `data_processor.py`: Handles cleaning and feature engineering.
    - `model.py`: Contains the ML model architectures (ARIMA/Regression).
    - `utils.py`: Logic for interactive visualizations and reporting.
- `data/`: Contains the source dataset (`train.csv`).
- `outputs/`: Directory for saved visualizations and business reports.
- `.streamlit/`: Configuration for the custom professional dashboard theme.

## 💻 Technical Stack
- **Language**: Python 3.x
- **Frameworks**: Streamlit, Scikit-learn
- **Data Analysis**: Pandas, NumPy
- **Statistical Modeling**: Statsmodels (ARIMA)
- **Visualizations**: Plotly

**Installation**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sales-forecasting-ml.git
cd sales-forecasting-ml
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

4. **Open your browser**
```
http://localhost:8501
```
