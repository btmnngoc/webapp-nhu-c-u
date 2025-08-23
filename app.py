import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from pmdarima import auto_arima
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# Hàm tải và xử lý dữ liệu
def tai_va_xu_ly_du_lieu(file_path, ma_san_pham):
    try:
        don_hang = pd.read_excel(file_path, sheet_name='Danh mục đơn hàng giả định')
        sua_chua = pd.read_excel(file_path, sheet_name='Phiếu sửa chữa')
        
        if not pd.api.types.is_datetime64_any_dtype(don_hang['DocDate']):
            don_hang['DocDate'] = pd.to_datetime(don_hang['DocDate'], origin='1899-12-30', unit='D', errors='coerce')
        
        if pd.api.types.is_numeric_dtype(sua_chua['Ngày']):
            sua_chua['Date'] = pd.to_datetime(sua_chua['Ngày'], origin='1899-12-30', unit='D', errors='coerce')
        else:
            sua_chua['Date'] = pd.to_datetime(sua_chua['Ngày'], errors='coerce', dayfirst=True)
        
        if ma_san_pham not in don_hang['ItemCode'].values and ma_san_pham not in sua_chua['Mã PT'].values:
            st.error(f"Mã sản phẩm {ma_san_pham} không tồn tại trong dữ liệu.")
            return None, None
            
        return don_hang, sua_chua
    except Exception as e:
        st.error(f"Lỗi load dữ liệu: {str(e)}")
        return None, None

# Hàm tổng hợp nhu cầu theo tuần
def tong_hop_nhu_cau(don_hang, sua_chua, ma_san_pham):
    don_hang_sp = don_hang[don_hang['ItemCode'] == ma_san_pham][['DocDate', 'Quantity', 'OrderFormName', 'Nhóm ĐL']].copy()
    sua_chua_sp = sua_chua[sua_chua['Mã PT'] == ma_san_pham][['Date', 'Slg']].copy()

    don_hang_sp['Week'] = don_hang_sp['DocDate'].dt.to_period('W').apply(lambda r: r.start_time)
    sua_chua_sp['Week'] = sua_chua_sp['Date'].dt.to_period('W').apply(lambda r: r.start_time)

    don_hang_tuan = don_hang_sp.groupby('Week').agg({
        'Quantity': 'sum',
        'OrderFormName': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'Nhóm ĐL': lambda x: x.mode()[0] if not x.empty else 'Unknown'
    }).reset_index()
    sua_chua_tuan = sua_chua_sp.groupby('Week')['Slg'].sum().reset_index()
    sua_chua_tuan.rename(columns={'Slg': 'Quantity'}, inplace=True)

    nhu_cau_tuan = pd.merge(don_hang_tuan, sua_chua_tuan, on='Week', how='outer', suffixes=('_don_hang', '_sua_chua'))
    nhu_cau_tuan['y'] = pd.to_numeric(nhu_cau_tuan['Quantity_don_hang'].fillna(0) + nhu_cau_tuan['Quantity_sua_chua'].fillna(0), errors='coerce').fillna(0)
    
    min_date = nhu_cau_tuan['Week'].min()
    max_date = pd.Timestamp('2025-08-07')
    all_weeks = pd.date_range(start=min_date, end=max_date, freq='W-MON').to_frame(index=False, name='Week')
    nhu_cau_tuan = pd.merge(all_weeks, nhu_cau_tuan, on='Week', how='left')
    nhu_cau_tuan['y'] = pd.to_numeric(nhu_cau_tuan['y'], errors='coerce').fillna(0)
    nhu_cau_tuan['OrderFormName'] = nhu_cau_tuan['OrderFormName'].fillna('Unknown')
    nhu_cau_tuan['Nhóm ĐL'] = nhu_cau_tuan['Nhóm ĐL'].fillna('Unknown')
    
    for lag in [1, 2, 3, 4, 12]:
        nhu_cau_tuan[f'lag_{lag}'] = nhu_cau_tuan['y'].shift(lag).fillna(0).astype(float)
    nhu_cau_tuan['non_zero'] = (nhu_cau_tuan['y'] > 0).astype(int)
    nhu_cau_tuan['rolling_mean'] = nhu_cau_tuan['y'].rolling(window=4).mean().fillna(0).astype(float)
    nhu_cau_tuan['rolling_std'] = nhu_cau_tuan['y'].rolling(window=4).std().fillna(0).astype(float)
    nhu_cau_tuan['week_of_year'] = nhu_cau_tuan['Week'].dt.isocalendar().week
    nhu_cau_tuan['seasonal_index'] = np.sin(2 * np.pi * nhu_cau_tuan['week_of_year'] / 52)
    nhu_cau_tuan['peak_week'] = nhu_cau_tuan['Week'].dt.isocalendar().week.isin([1, 2, 3, 4, 48, 49, 50, 51, 52]).astype(int)
    
    nhu_cau_tuan['y_log'] = np.log1p(nhu_cau_tuan['y'] + 0.1)
    
    return nhu_cau_tuan

# Streamlit app
st.set_page_config(page_title="Dự Báo Nhu Cầu Xuất Kho Phụ Tùng", layout="wide")
st.title("Dự Báo Nhu Cầu Xuất Kho Phụ Tùng")
st.markdown("Ứng dụng dự báo nhu cầu xuất kho phụ tùng theo tuần sử dụng nhiều mô hình học máy.")

# Nhập mã sản phẩm và chọn mô hình
st.header("Nhập mã sản phẩm và chọn mô hình")
col1, col2 = st.columns([1, 1])
with col1:
    ma_san_pham = st.text_input("Mã sản phẩm (ví dụ: 263304A001)", value="263304A001")
with col2:
    model_choice = st.selectbox("Chọn mô hình", ["XGBoost", "Random Forest", "ARIMA", "Prophet"])

# Đường dẫn file cố định
file_path = 'data/2025.01. Dữ liệu giả định.xlsx'

if ma_san_pham:
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Đặc trưng", "Tham số tối ưu", "Đánh giá mô hình", "Dự báo 24 tuần"])

    # Tab 1: Đặc trưng
    with tab1:
        st.header("Bảng đặc trưng")
        try:
            don_hang, sua_chua = tai_va_xu_ly_du_lieu(file_path, ma_san_pham)
            if don_hang is None or sua_chua is None:
                st.error("Lỗi load dữ liệu. Vui lòng kiểm tra file Excel trong thư mục data.")
            else:
                df = tong_hop_nhu_cau(don_hang, sua_chua, ma_san_pham)
                df['ItemCode'] = ma_san_pham
                st.dataframe(df[['Week', 'y', 'y_log', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_12', 'non_zero', 
                                 'rolling_mean', 'rolling_std', 'week_of_year', 'seasonal_index', 'peak_week', 
                                 'OrderFormName', 'Nhóm ĐL']].style.format({
                                     'y': '{:.2f}', 'y_log': '{:.2f}', 'lag_1': '{:.2f}', 'lag_2': '{:.2f}', 
                                     'lag_3': '{:.2f}', 'lag_4': '{:.2f}', 'lag_12': '{:.2f}', 
                                     'rolling_mean': '{:.2f}', 'rolling_std': '{:.2f}', 'seasonal_index': '{:.2f}'
                                 }), height=300)
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")

    # Tab 2: Tham số tối ưu
    with tab2:
        if 'df' in locals():
            st.header("Tham số tối ưu")
            df = pd.get_dummies(df, columns=['OrderFormName', 'Nhóm ĐL'], prefix=['OrderFormName', 'Nhóm ĐL'])
            feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_12', 'non_zero', 'rolling_mean', 
                            'rolling_std', 'week_of_year', 'seasonal_index', 'peak_week'] + \
                           [col for col in df.columns if col.startswith('OrderFormName_') or col.startswith('Nhóm ĐL_')]
            feature_cols = [col for col in feature_cols if df[col].var() > 0.01]

            X = df[feature_cols].astype(float)
            y = df['y_log'].astype(float)

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            if model_choice in ["XGBoost", "Random Forest"]:
                if model_choice == "XGBoost":
                    model = XGBRegressor(objective='reg:squarederror')
                    param_grid = {
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'n_estimators': [100, 200, 300],
                        'min_child_weight': [1, 2]
                    }
                else:  # Random Forest
                    model = RandomForestRegressor(random_state=42)
                    param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5]
                    }
                
                tscv = TimeSeriesSplit(n_splits=3)
                grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=tscv, verbose=0)
                grid_search.fit(X_scaled, y)
                best_params = grid_search.best_params_
                st.write(f"Tham số tối ưu cho {model_choice}: {best_params}")
            
            elif model_choice == "ARIMA":
                model = auto_arima(df['y'], seasonal=True, m=52, suppress_warnings=True)
                st.write(f"Tham số tối ưu cho ARIMA: {model.order}, Seasonal: {model.seasonal_order}")
            
            else:  # Prophet
                st.write("Prophet: Sử dụng cấu hình mặc định với mùa vụ hàng tuần.")
        else:
            st.error("Vui lòng kiểm tra dữ liệu ở tab Đặc trưng trước.")

    # Tab 3: Đánh giá mô hình
    with tab3:
        if 'X_scaled' in locals() and 'y' in locals() and 'df' in locals():
            st.header("Đánh giá mô hình (Sliding Window)")
            target_iterations = 30
            kich_thuoc_test = 24  # 6 tháng ~24 tuần
            all_metrics = []
            all_test_plots = []
            all_error_plots = []

            total_weeks = len(df)
            max_iterations = total_weeks - kich_thuoc_test - 1
            iterations = min(target_iterations, max_iterations)

            for i in range(iterations):
                end_test_idx = total_weeks - i
                start_test_idx = end_test_idx - kich_thuoc_test
                y_test = df['y_log'].iloc[start_test_idx:end_test_idx]
                
                if model_choice in ["XGBoost", "Random Forest"]:
                    X_test = X_scaled[start_test_idx:end_test_idx]
                    X_train = X_scaled[:start_test_idx]
                    y_train = y[:start_test_idx]
                    
                    if len(X_train) < 1:
                        st.warning(f"Iteration {i}: Không đủ dữ liệu huấn luyện.")
                        break
                    
                    if model_choice == "XGBoost":
                        model = XGBRegressor(objective='reg:squarederror', **best_params)
                    else:
                        model = RandomForestRegressor(random_state=42, **best_params)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                elif model_choice == "ARIMA":
                    train_data = df['y'].iloc[:start_test_idx]
                    if len(train_data) < 1:
                        st.warning(f"Iteration {i}: Không đủ dữ liệu huấn luyện.")
                        break
                    model = auto_arima(train_data, seasonal=True, m=52, suppress_warnings=True)
                    y_pred = model.predict(n_periods=kich_thuoc_test)
                    y_pred = np.log1p(np.clip(y_pred, 0, 4000) + 0.1)
                
                else:  # Prophet
                    train_data = df[['Week', 'y']].iloc[:start_test_idx].rename(columns={'Week': 'ds', 'y': 'y'})
                    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
                    model.fit(train_data)
                    future = model.make_future_dataframe(periods=kich_thuoc_test, freq='W-MON')
                    forecast = model.predict(future)
                    y_pred = np.log1p(np.clip(forecast['yhat'].iloc[-kich_thuoc_test:], 0, 4000) + 0.1)
                
                y_pred = np.clip(np.expm1(y_pred), 0, 4000)
                y_test = np.expm1(y_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae_percent = (mae / np.mean(y_test) * 100) if np.mean(y_test) > 0 else np.nan
                rmse_percent = (rmse / np.mean(y_test) * 100) if np.mean(y_test) > 0 else np.nan
                
                metrics = {
                    'Iteration': i,
                    'Test_Start': df['Week'].iloc[start_test_idx],
                    'Test_End': df['Week'].iloc[end_test_idx-1],
                    'MAE': mae,
                    'MAE_%': mae_percent,
                    'RMSE': rmse,
                    'RMSE_%': rmse_percent
                }
                all_metrics.append(metrics)
                
                test_plot_data = pd.DataFrame({
                    'Week': df['Week'].iloc[start_test_idx:end_test_idx],
                    'Thực tế': y_test,
                    'Dự báo': y_pred
                })
                errors = y_test - y_pred
                all_test_plots.append((i, test_plot_data))
                all_error_plots.append((i, errors))

            # Bảng metrics
            st.subheader("Bảng đánh giá (MAE, MAE%, RMSE, RMSE%)")
            all_metrics_df = pd.DataFrame(all_metrics)
            all_metrics_df['Test_Start'] = all_metrics_df['Test_Start'].dt.strftime('%Y-%m-%d')
            all_metrics_df['Test_End'] = all_metrics_df['Test_End'].dt.strftime('%Y-%m-%d')
            st.dataframe(all_metrics_df.style.format({
                'MAE': '{:.2f}',
                'MAE_%': '{:.2f}%',
                'RMSE': '{:.2f}',
                'RMSE_%': '{:.2f}%'
            }), height=300)

            # Thống kê metrics
            st.subheader("Thống kê sai số")
            st.write(f"Mean MAE%: {all_metrics_df['MAE_%'].mean():.2f}%")
            st.write(f"Std MAE%: {all_metrics_df['MAE_%'].std():.2f}%")
            st.write(f"Mean RMSE%: {all_metrics_df['RMSE_%'].mean():.2f}%")
            st.write(f"Std RMSE%: {all_metrics_df['RMSE_%'].std():.2f}%")

            # Đồ thị dự báo và sai số
            st.subheader("Đồ thị dự báo và sai số")
            selected_iteration = st.selectbox("Chọn Iteration", [i for i, _ in all_test_plots], key="eval_select")
            
            test_plot_data = next(plot for i, plot in all_test_plots if i == selected_iteration)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_plot_data['Week'], y=test_plot_data['Thực tế'], 
                                     mode='lines+markers', name='Thực tế', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_plot_data['Week'], y=test_plot_data['Dự báo'], 
                                     mode='lines+markers', name='Dự báo', line=dict(color='orange', dash='dash')))
            fig.update_layout(
                title=f"Dự báo trên tập test (Iteration {selected_iteration})",
                xaxis_title="Tuần",
                yaxis_title="Số lượng xuất kho",
                template="plotly_white",
                showlegend=True,
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig, use_container_width=True)

            errors = next(err for i, err in all_error_plots if i == selected_iteration)
            fig = px.histogram(x=errors, nbins=20, title=f"Phân phối sai số (Iteration {selected_iteration})")
            fig.update_layout(
                xaxis_title="Sai số (Thực tế - Dự báo)",
                yaxis_title="Tần suất",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Vui lòng kiểm tra dữ liệu ở tab Đặc trưng và Tham số tối ưu trước.")

    # Tab 4: Dự báo 24 tuần
    with tab4:
        if 'X_scaled' in locals() and 'y' in locals() and 'df' in locals():
            st.header("Dự báo 24 tuần tiếp theo (từ 11/08/2025)")
            
            future_dates = pd.date_range(start='2025-08-11', periods=24, freq='W-MON')
            if model_choice in ["XGBoost", "Random Forest"]:
                model = XGBRegressor(objective='reg:squarederror', **best_params) if model_choice == "XGBoost" else RandomForestRegressor(random_state=42, **best_params)
                model.fit(X_scaled, y)
                
                future_df = pd.DataFrame(index=range(24))
                last_row = df.iloc[-1]
                last_y = df['y_log'].iloc[-12:].values
                
                for col in feature_cols:
                    if col in ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_12']:
                        future_df[col] = 0
                    elif col == 'non_zero':
                        future_df[col] = 1
                    elif col == 'rolling_mean':
                        future_df[col] = last_row['rolling_mean']
                    elif col == 'rolling_std':
                        future_df[col] = last_row['rolling_std']
                    elif col == 'week_of_year':
                        future_df[col] = [pd.Timestamp(d).isocalendar().week for d in future_dates]
                    elif col == 'seasonal_index':
                        future_df[col] = [np.sin(2 * np.pi * pd.Timestamp(d).isocalendar().week / 52) for d in future_dates]
                    elif col == 'peak_week':
                        future_df[col] = [1 if pd.Timestamp(d).isocalendar().week in [1, 2, 3, 4, 48, 49, 50, 51, 52] else 0 for d in future_dates]
                    else:
                        future_df[col] = last_row[col]

                forecast = []
                for i in range(24):
                    future_df.loc[i, 'lag_1'] = last_y[-1]
                    future_df.loc[i, 'lag_2'] = last_y[-2] if len(last_y) > 1 else 0
                    future_df.loc[i, 'lag_3'] = last_y[-3] if len(last_y) > 2 else 0
                    future_df.loc[i, 'lag_4'] = last_y[-4] if len(last_y) > 3 else 0
                    future_df.loc[i, 'lag_12'] = last_y[-12] if len(last_y) > 11 else 0
                    X_future = scaler.transform(future_df.iloc[[i]][feature_cols])
                    pred = model.predict(X_future)
                    forecast.append(np.clip(np.expm1(pred[0]), 0, 4000))
                    last_y = np.append(last_y, pred)
            
            elif model_choice == "ARIMA":
                model = auto_arima(df['y'], seasonal=True, m=52, suppress_warnings=True)
                forecast = model.predict(n_periods=24)
                forecast = np.clip(forecast, 0, 4000)
            
            else:  # Prophet
                prophet_df = df[['Week', 'y']].rename(columns={'Week': 'ds', 'y': 'y'})
                model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=24, freq='W-MON')
                forecast_df = model.predict(future)
                forecast = np.clip(forecast_df['yhat'].iloc[-24:], 0, 4000)

            df_future = pd.DataFrame({
                'Week': future_dates,
                'Quantity': forecast
            })
            df_future['Week'] = df_future['Week'].dt.strftime('%Y-%m-%d')
            st.subheader("Bảng dự báo 24 tuần")
            st.dataframe(df_future.style.format({'Quantity': '{:.2f}'}), height=300)

            historical_data = pd.DataFrame({
                'Week': df['Week'],
                'Quantity': np.expm1(df['y_log'])
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=historical_data['Week'], y=historical_data['Quantity'], 
                                     mode='lines+markers', name='Thực tế', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df_future['Week'], y=df_future['Quantity'], 
                                     mode='lines+markers', name='Dự báo', line=dict(color='orange', dash='dash')))
            fig.update_layout(
                title="Dự báo nhu cầu xuất kho 24 tuần tiếp theo (từ 11/08/2025)",
                xaxis_title="Tuần",
                yaxis_title="Số lượng xuất kho",
                template="plotly_white",
                showlegend=True,
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Vui lòng kiểm tra dữ liệu ở các tab trước.")
else:
    st.info("Vui lòng nhập mã sản phẩm và chọn mô hình để bắt đầu.")