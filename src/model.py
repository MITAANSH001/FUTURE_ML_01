import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class SalesForecaster:
    def train_arima(self, ts_data):
        # HARD GUARD: ARIMA usually needs at least 10-20 points for stability
        # We will use 5 as the absolute minimum to avoid the IndexError
        if len(ts_data) < 5:
            # Return dummy metrics so the dashboard doesn't crash
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'status': "Insufficient Data (Need 5+ days)"
            }
        
        try:
            model = ARIMA(ts_data, order=(1, 0, 0)).fit()
            predictions = model.fittedvalues
            return {
                'mae': mean_absolute_error(ts_data, predictions),
                'rmse': np.sqrt(mean_squared_error(ts_data, predictions)),
                'mape': np.mean(np.abs((ts_data - predictions) / ts_data.replace(0, 1))) * 100,
                'status': "ARIMA Model Active"
            }
        except Exception as e:
            return {'mae': 0, 'rmse': 0, 'mape': 0, 'status': f"Model Error: {str(e)}"}

    def predict_future(self, ts_data, steps=7):
        if len(ts_data) < 5:
            # Fallback: Just repeat the last known value
            last_val = ts_data.iloc[-1]
            future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
            return pd.Series([last_val] * steps, index=future_dates)
        
        model = ARIMA(ts_data, order=(1, 0, 0)).fit()
        return model.forecast(steps=steps)
