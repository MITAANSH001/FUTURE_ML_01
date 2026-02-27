from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

class SalesForecaster:
    def __init__(self):
        self.model = None
        self.arima_model = None

    def _get_mape(self, y_true, y_pred):
        """Helper to calculate MAPE safely."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    def train_regression(self, df_features):
        X = df_features[['time_index', 'month', 'lag_1', 'lag_12']]
        y = df_features['sales']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        return {
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'mae': round(mean_absolute_error(y_test, y_pred), 2),
            'mape': round(self._get_mape(y_test, y_pred), 2),
            'predictions': y_pred,
            'actuals': y_test
        }

    def train_arima(self, ts_data):
        split_idx = int(len(ts_data) * 0.8)
        train_data = ts_data.iloc[:split_idx]
        test_data = ts_data.iloc[split_idx:]
        
        # Fitting with basic order (1,1,1)
        self.arima_model = ARIMA(train_data, order=(1,1,1)).fit()
        forecast = self.arima_model.forecast(steps=len(test_data))
        
        return {
            'rmse': round(np.sqrt(mean_squared_error(test_data, forecast)), 2),
            'mae': round(mean_absolute_error(test_data, forecast), 2),
            'mape': round(self._get_mape(test_data, forecast), 2),
            'predictions': forecast,
            'actuals': test_data
        }

    def predict_future(self, last_data, steps=12):
        # Full model training for final forecast
        full_model = ARIMA(last_data, order=(1,1,1)).fit()
        return full_model.forecast(steps=steps)