import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads and sorts the raw data."""
        try:
            self.df = pd.read_csv(self.file_path)
            # Standardize date and sort for time-series integrity
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date')
            self.df = self.df.fillna(0)
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def get_time_series_data(self, freq='MS'):
        """Groups data by date and resamples to a fixed frequency (Month Start)."""
        if self.df is None:
            self.load_data()
        
        # Aggregate sales by date
        ts_data = self.df.groupby('date')['sales'].sum().reset_index()
        ts_data.set_index('date', inplace=True)
        
        # Resample ensures no missing months in the timeline
        ts_data = ts_data['sales'].resample(freq).sum()
        return ts_data

    def engineer_features(self, ts_data):
        """Creates lag features and time indices for the ML model."""
        df_features = ts_data.reset_index()
        df_features.columns = ['date', 'sales']
        
        # Time components
        df_features['month'] = df_features['date'].dt.month
        df_features['year'] = df_features['date'].dt.year
        df_features['time_index'] = np.arange(len(df_features))
        
        # Lag features: Critical for AI/ML forecasting
        df_features['lag_1'] = df_features['sales'].shift(1)
        df_features['lag_12'] = df_features['sales'].shift(12)
        
        # Drop NaN values created by shifts
        df_features = df_features.dropna()
        return df_features