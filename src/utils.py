import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

class Visualizer:
    def __init__(self, output_dir='outputs/visualizations/'):
        self.output_dir = output_dir
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_sales_trend(self, ts_data):
        """Creates an interactive historical sales trend line chart."""
        fig = px.line(
            ts_data, 
            x=ts_data.index, 
            y='sales', 
            title='Monthly Sales Trend Analysis',
            labels={'sales': 'Total Sales ($)', 'date': 'Timeline'},
            template="plotly_dark" # Matches your dark theme
        )
        fig.update_traces(line_color='#00d4ff', mode='lines+markers')
        return fig

    def plot_forecast_vs_actual(self, actual, predicted):
        """Compares actual vs forecasted values with a dual-line plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=actual.index, y=actual, name='Actual Sales',
                                 line=dict(color='#00d4ff', width=3)))
        
        fig.add_trace(go.Scatter(x=actual.index, y=predicted, name='Forecasted Sales',
                                 line=dict(color='#ff4b4b', dash='dash')))
        
        fig.update_layout(title='Model Accuracy: Forecast vs Actual',
                          xaxis_title='Date', yaxis_title='Sales ($)',
                          template="plotly_dark")
        return fig

    def plot_future_forecast(self, historical, future):
        """Visualizes the 12-month future prediction horizon."""
        # Generate future dates for the X-axis
        last_date = historical.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future)+1, freq='MS')[1:]
        
        fig = go.Figure()
        
        # Historical Data (Last 24 months for context)
        hist_context = historical.tail(24)
        fig.add_trace(go.Scatter(x=hist_context.index, y=hist_context, 
                                 name='Historical Context', line=dict(color='gray')))
        
        # Future Prediction
        fig.add_trace(go.Scatter(x=future_dates, y=future, 
                                 name='Predicted Future', line=dict(color='#00d4ff', width=4)))
        
        fig.update_layout(title='12-Month Projected Sales Forecast',
                          template="plotly_dark", hovermode="x unified")
        return fig

    def generate_business_summary(self, future_forecast, metrics, filename='outputs/report.md'):
        """Generates the Markdown report for business stakeholders."""
        total_sales = future_forecast.sum()