from pmdarima import auto_arima
import numpy as np

def train_arima(train_data, kich_thuoc_test):
    model = auto_arima(train_data, seasonal=True, m=52, suppress_warnings=True)  # m=52 cho tuần
    return model

def predict_arima(model, kich_thuoc_test):
    y_pred = model.predict(n_periods=kich_thuoc_test)
    return np.log1p(np.clip(y_pred, 0, 4000) + 0.1)