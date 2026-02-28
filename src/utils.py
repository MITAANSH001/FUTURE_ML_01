import plotly.graph_objects as go

class Visualizer:
    def plot_future_forecast(self, historical, forecast):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical.index, y=historical.values, name="Historical"))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast", line=dict(dash='dash')))
        fig.update_layout(title="Sales Forecast", template="plotly_dark")
        return fig
