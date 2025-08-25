from prophet import Prophet
import numpy as np

def train_prophet(train_data, period_col):
    model = Prophet(yearly_seasonality=True)
    model.fit(train_data.rename(columns={period_col: 'ds', 'y': 'y'}))
    return model

def predict_prophet(model, kich_thuoc_test, period_col, granularity):
    future = model.make_future_dataframe(periods=kich_thuoc_test, freq='W-MON' if granularity == 'W' else 'MS')
    forecast = model.predict(future)
    return np.log1p(np.clip(forecast['yhat'].iloc[-kich_thuoc_test:], 0, 4000) + 0.1)