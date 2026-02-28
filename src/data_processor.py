import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        # Use a single tab or 8 spaces to match the line above
        self.df['date'] = pd.to_datetime(self.df['date'])
        return self.df

    def get_time_series_data(self):
        if self.df is None:
            self.load_data()
        ts_data = self.df.groupby('date')['sales'].sum().sort_index()
        ts_data = ts_data.asfreq('D').fillna(0)
        return ts_data
