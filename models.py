import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pmdarima import auto_arima
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

def create_dense_model(input_size, number_layers=0):
    print("create Dense model ")
    model = Sequential()
    model.add(Dense(input_size, activation='relu', input_shape=(input_size,)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    print('Compiling...')
    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(0.001),
                  metrics=['mean_squared_error', 'RootMeanSquaredError'])
    return model

def train_dense(model, X_train, y_train):
    model.fit(X_train, y_train, batch_size=50, epochs=200, validation_split=0.2, verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)])
    return model

def testing_dense(model, X_test):
    y_predicted = model.predict(X_test, verbose=0)
    return y_predicted.flatten()

def get_best_params(model_choice, X_scaled, y, feature_cols):
    if model_choice in ["XGBoost", "Random Forest"]:
        if model_choice == "XGBoost":
            model = XGBRegressor(objective='reg:squarederror')
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 2]
            }
        else:
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            }
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=tscv, verbose=0)
        grid_search.fit(X_scaled, y)
        return grid_search.best_params_
    elif model_choice == "ARIMA":
        model = auto_arima(y, seasonal=True, m=12, suppress_warnings=True)
        return {"order": model.order, "seasonal_order": model.seasonal_order}
    elif model_choice == "Prophet":
        return "Cấu hình mặc định với mùa vụ hàng năm."
    elif model_choice == "Neural Network":
        return f"Cấu trúc - Input: {len(feature_cols)} (ReLU), Hidden: 32 (sigmoid), Dropout: 0.2, Output: 1 (linear), Optimizer: Adam, Loss: MAE"
    return None

def evaluate_model(model_choice, df, X_scaled, y, feature_cols, scaler, best_params, period_col):
    kich_thuoc_test = 24  # Cố định 24 tuần cho tất cả mô hình
    target_iterations = 10
    all_metrics = []
    all_test_plots = []
    all_error_plots = []

    total_periods = len(df)
    max_iterations = total_periods - kich_thuoc_test - 1
    iterations = min(target_iterations, max_iterations)

    if total_periods < kich_thuoc_test + 12:
        st.warning(f"Dữ liệu quá ngắn để huấn luyện với tập test {kich_thuoc_test} tuần. Cần ít nhất {kich_thuoc_test + 12} tuần.")
        return None, None, None

    for i in range(iterations):
        end_test_idx = total_periods - i
        start_test_idx = end_test_idx - kich_thuoc_test
        y_test = df['y_log'].iloc[start_test_idx:end_test_idx]
        X_test = X_scaled[start_test_idx:end_test_idx]
        X_train = X_scaled[:start_test_idx]
        y_train = y[:start_test_idx]
        
        if len(X_train) < 12:
            continue
        
        if model_choice in ["XGBoost", "Random Forest", "Neural Network"]:
            if model_choice == "XGBoost":
                model = XGBRegressor(objective='reg:squarederror', **best_params)
            elif model_choice == "Random Forest":
                model = RandomForestRegressor(random_state=42, **best_params)
            else:
                model = create_dense_model(X_train.shape[1])
                model = train_dense(model, X_train, y_train)
            if model_choice != "Neural Network":
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test) if model_choice != "Neural Network" else testing_dense(model, X_test)
        
        elif model_choice == "ARIMA":
            train_data = df['y'].iloc[:start_test_idx]
            model = auto_arima(train_data, seasonal=True, m=52, suppress_warnings=True)  # m=52 cho tuần
            y_pred = model.predict(n_periods=kich_thuoc_test)
            y_pred = np.log1p(np.clip(y_pred, 0, 4000) + 0.1)
        
        else:  # Prophet
            train_data = df[[period_col, 'y']].iloc[:start_test_idx].rename(columns={period_col: 'ds', 'y': 'y'})
            model = Prophet(yearly_seasonality=True)
            model.fit(train_data)
            future = model.make_future_dataframe(periods=kich_thuoc_test, freq='W-MON')
            forecast = model.predict(future)
            y_pred = np.log1p(np.clip(forecast['yhat'].iloc[-kich_thuoc_test:], 0, 4000) + 0.1)
        
        y_pred = np.clip(np.expm1(y_pred), 0, 4000)
        y_test = np.expm1(y_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        mae_percent = (mae / np.mean(y_test) * 100) if np.mean(y_test) > 0 else np.nan
        rmse_percent = (rmse / np.mean(y_test) * 100) if np.mean(y_test) > 0 else np.nan
        
        metrics = {
            'Iteration': i,
            'Test_Start': df[period_col].iloc[start_test_idx],
            'Test_End': df[period_col].iloc[end_test_idx-1],
            'MAE': mae,
            'MAE_%': mae_percent,
            'RMSE': rmse,
            'RMSE_%': rmse_percent,
            'MAPE_%': mape
        }
        all_metrics.append(metrics)
        
        test_plot_data = pd.DataFrame({
            'Iteration': [i],
            'Period': [df[period_col].iloc[start_test_idx:end_test_idx]],
            'Thực tế': [y_test],
            'Dự báo': [y_pred]
        })
        errors = y_test - y_pred
        all_test_plots.append(test_plot_data)
        all_error_plots.append(pd.DataFrame({'Iteration': [i], 'errors': [errors]}))

    all_metrics_df = pd.DataFrame(all_metrics)
    top_10_metrics = all_metrics_df[all_metrics_df['MAE_%'].notna()].sort_values('MAE_%').head(10)
    top_10_test_plots = pd.concat([plot for plot in all_test_plots if plot['Iteration'].values[0] in top_10_metrics['Iteration'].tolist()])
    top_10_error_plots = pd.concat([err for err in all_error_plots if err['Iteration'].values[0] in top_10_metrics['Iteration'].tolist()])
    
    return top_10_metrics, top_10_test_plots, top_10_error_plots

def forecast_future_demand(model_choice, df, feature_cols, scaler, future_dates, best_params, granularity, period_col):
    if model_choice in ["XGBoost", "Random Forest", "Neural Network"]:
        y = df['y_log'].astype(float)
        X_scaled = scaler.transform(df[feature_cols])
        if model_choice == "XGBoost":
            model = XGBRegressor(objective='reg:squarederror', **best_params)
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(random_state=42, **best_params)
        else:
            model = create_dense_model(X_scaled.shape[1])
            model = train_dense(model, X_scaled, y)
        model.fit(X_scaled, y) if model_choice != "Neural Network" else None
        
        future_df = pd.DataFrame(index=range(len(future_dates)))
        last_row = df.iloc[-1]
        last_y = df['y_log'].iloc[-12:].values  # Dùng lag_12 cho cả tháng và tuần
        
        date_col = 'month_of_year' if granularity == 'M' else 'week_of_year'
        lags = [1, 2, 3, 4, 12]
        seasonal_div = 12 if granularity == 'M' else 52
        
        for col in feature_cols:
            if col in [f'lag_{lag}' for lag in lags]:
                future_df[col] = 0
            elif col == 'non_zero':
                future_df[col] = 1
            elif col == 'rolling_mean':
                future_df[col] = last_row['rolling_mean']
            elif col == 'rolling_std':
                future_df[col] = last_row['rolling_std']
            elif col == 'Quantity_don_hang':
                future_df[col] = last_row['Quantity_don_hang']
            elif col == 'Quantity_sua_chua':
                future_df[col] = last_row['Quantity_sua_chua']
            elif col == date_col:
                if granularity == 'M':
                    future_df[col] = [pd.Timestamp(d).month for d in future_dates]
                else:
                    future_df[col] = [pd.Timestamp(d).isocalendar()[1] for d in future_dates]
            elif col == 'seasonal_index':
                if granularity == 'M':
                    future_df[col] = [np.sin(2 * np.pi * pd.Timestamp(d).month / seasonal_div) for d in future_dates]
                else:
                    future_df[col] = [np.sin(2 * np.pi * pd.Timestamp(d).isocalendar()[1] / seasonal_div) for d in future_dates]
            elif col == 'peak_period':
                if granularity == 'M':
                    future_df[col] = [1 if pd.Timestamp(d).month in [1, 2, 11, 12] else 0 for d in future_dates]
                else:
                    future_df[col] = [1 if pd.Timestamp(d).isocalendar()[1] in [1, 52] else 0 for d in future_dates]
            else:
                future_df[col] = last_row[col]

        forecast = []
        for i in range(len(future_dates)):
            future_df.loc[i, f'lag_1'] = last_y[-1]
            future_df.loc[i, f'lag_2'] = last_y[-2] if len(last_y) > 1 else 0
            future_df.loc[i, f'lag_3'] = last_y[-3] if len(last_y) > 2 else 0
            future_df.loc[i, f'lag_4'] = last_y[-4] if len(last_y) > 3 else 0
            future_df.loc[i, f'lag_12'] = last_y[-12] if len(last_y) > 11 else 0
            X_future = scaler.transform(future_df.iloc[[i]][feature_cols])
            pred = model.predict(X_future)
            pred = pred[0] if model_choice != "Neural Network" else testing_dense(model, X_future)[0]
            forecast.append(np.clip(np.expm1(pred), 0, 4000))
            last_y = np.append(last_y, pred)
    
    elif model_choice == "ARIMA":
        model = auto_arima(df['y'], seasonal=True, m=12 if granularity == 'M' else 52, suppress_warnings=True)
        forecast = model.predict(n_periods=len(future_dates))
        forecast = np.clip(forecast, 0, 4000)
    
    elif model_choice == "Prophet":
        prophet_df = df[[period_col, 'y']].rename(columns={period_col: 'ds', 'y': 'y'})
        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=len(future_dates), freq='MS' if granularity == 'M' else 'W-MON')
        forecast_df = model.predict(future)
        forecast = np.clip(forecast_df['yhat'].iloc[-len(future_dates):], 0, 4000)
    
    df_future = pd.DataFrame({
        period_col: future_dates,
        'Quantity': forecast
    })
    df_future[period_col] = df_future[period_col].dt.strftime('%Y-%m-%d')
    return df_future