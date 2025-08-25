import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from data_utils import tai_va_xu_ly_du_lieu, tong_hop_nhu_cau
from model_utils import get_best_params, evaluate_model, forecast_future_demand
from visualizations import display_features_table, display_metrics_table, display_stats, plot_forecast_vs_actual, plot_error_histogram, plot_historical_and_forecast, display_actual_vs_forecast_table

st.set_page_config(page_title="Dự Báo Nhu Cầu Xuất Kho Phụ Tùng", layout="wide")
st.title("Dự Báo Nhu Cầu Xuất Kho Phụ Tùng")
st.markdown("Ứng dụng dự báo nhu cầu xuất kho phụ tùng theo tháng hoặc tuần sử dụng nhiều mô hình học máy.")

def aggregate_week_to_month(df, period_col):
    df = df.copy()
    df['Month'] = pd.to_datetime(df[period_col]).dt.to_period('M').dt.to_timestamp()
    df_month = df.groupby('Month').agg({'Quantity': 'sum'}).reset_index()
    return df_month

# Nhập mã sản phẩm, chọn mô hình và kỳ dự báo
st.header("Nhập mã sản phẩm, chọn mô hình và kỳ dự báo")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    ma_san_pham = st.text_input("Mã sản phẩm (ví dụ: 263304A001)", value="263304A001")
with col2:
    model_choice = st.selectbox("Chọn mô hình", ["XGBoost", "Random Forest", "ARIMA", "Prophet", "Neural Network"])
with col3:
    granularity_choice = st.selectbox("Chọn kỳ dự báo", ["Theo tháng", "Theo tuần"])
    granularity = 'M' if granularity_choice == "Theo tháng" else 'W'

# Đường dẫn file cố định
file_path = 'data/2025.01. Dữ liệu giả định.xlsx'

# Cảnh báo dữ liệu ngắn
if granularity == 'W':
    don_hang, sua_chua = tai_va_xu_ly_du_lieu(file_path)
    if don_hang is not None and sua_chua is not None:
        df, _ = tong_hop_nhu_cau(don_hang, sua_chua, ma_san_pham, granularity)
        if df is not None and len(df) < 36:
            st.warning("Dữ liệu tuần quá ngắn (< 36 tuần), kết quả Neural Network có thể không ổn định. Cần ít nhất 36 tuần để huấn luyện tốt.")
        elif len(df) < 24:
            st.error("Dữ liệu tuần quá ngắn (< 24 tuần), không đủ để đánh giá mô hình.")
            st.stop()

if ma_san_pham:
    don_hang, sua_chua = tai_va_xu_ly_du_lieu(file_path)
    if don_hang is None or sua_chua is None:
        st.error("Lỗi load dữ liệu. Vui lòng kiểm tra file Excel trong thư mục data.")
    else:
        df, period_col = tong_hop_nhu_cau(don_hang, sua_chua, ma_san_pham, granularity)
        if df is None:
            st.error("Không có dữ liệu cho mã sản phẩm này.")
        else:
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Đặc trưng", "Tham số tối ưu", "Đánh giá mô hình", "Dự báo 6 kỳ"])

            with tab1:
                st.header("Bảng đặc trưng")
                display_features_table(df, period_col)

            # Chuẩn bị features và scaling
            df_model = df.copy()
            df_model = pd.get_dummies(df_model, columns=['OrderFormName', 'Nhóm ĐL'], prefix=['OrderFormName', 'Nhóm ĐL'])
            date_col = 'month_of_year' if granularity == 'M' else 'week_of_year'
            lags = [1, 2, 3, 4, 12]
            feature_cols = [f'lag_{lag}' for lag in lags] + \
                           ['Quantity_don_hang', 'Quantity_sua_chua', 'non_zero', 'rolling_mean', 'rolling_std', 
                            date_col, 'seasonal_index', 'peak_period'] + \
                           [col for col in df_model.columns if col.startswith('OrderFormName_') or col.startswith('Nhóm ĐL_')]
            feature_cols = [col for col in feature_cols if col in df_model.columns and df_model[col].var() > 0.01]
            X = df_model[feature_cols].astype(float)
            y = df_model['y_log'].astype(float)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            with tab2:
                st.header("Tham số tối ưu")
                best_params = get_best_params(model_choice, X_scaled, y, feature_cols)
                if best_params:
                    st.write(f"Tham số tối ưu cho {model_choice}: {best_params}")

            with tab3:
                st.header("Đánh giá mô hình (Sliding Window - 10 lần)")
                top_10_metrics, top_10_test_plots, top_10_error_plots = evaluate_model(model_choice, df_model, X_scaled, y, feature_cols, scaler, best_params, period_col)
                if top_10_metrics is not None:
                    display_metrics_table(top_10_metrics)
                    display_stats(top_10_metrics)
                    selected_iteration = st.selectbox("Chọn Iteration", top_10_metrics['Iteration'].tolist(), key="eval_select")
                    test_plot_data = top_10_test_plots[top_10_test_plots['Iteration'] == selected_iteration].drop(columns='Iteration')
                    errors = top_10_error_plots[top_10_error_plots['Iteration'] == selected_iteration]['errors'].values[0]
                    plot_forecast_vs_actual(test_plot_data, selected_iteration, period_col)
                    plot_error_histogram(errors, selected_iteration, is_nn=(model_choice == "Neural Network"))

            with tab4:
                st.header(f"Dự báo 6 {granularity_choice.lower()} tiếp theo (từ {datetime(2025, 8, 25).strftime('%Y-%m-%d') if granularity == 'W' else '01/09/2025'})")
                start_date = '2025-08-25' if granularity == 'W' else '2025-09-01'
                freq = 'W-MON' if granularity == 'W' else 'MS'
                future_dates = pd.date_range(start=start_date, periods=6, freq=freq)
                df_future = forecast_future_demand(model_choice, df_model, feature_cols, scaler, future_dates, best_params, granularity, period_col)
                st.subheader(f"Bảng dự báo 6 {granularity_choice.lower()}")
                st.dataframe(df_future.style.format({'Quantity': '{:.2f}'}), height=300)
                historical_data = pd.DataFrame({period_col: df[period_col], 'Quantity': np.expm1(df['y_log'])})
                
                # Hiển thị bảng chuỗi thực tế và dự báo
                st.subheader(f"Bảng chuỗi thực tế và dự báo ({granularity_choice.lower()})")
                display_actual_vs_forecast_table(historical_data, df_future, period_col)
                
                # Đồ thị chuỗi thực tế và dự báo
                st.subheader(f"Đồ thị chuỗi thực tế và dự báo ({granularity_choice.lower()})")
                plot_historical_and_forecast(historical_data, df_future, period_col)

                # Nếu chọn tuần, hiển thị tổng hợp tháng
                if granularity == 'W':
                    st.subheader("Dữ liệu tổng hợp theo tháng (cộng 4 tuần)")
                    df_month_agg = aggregate_week_to_month(df_future, period_col)
                    st.dataframe(df_month_agg.style.format({'Quantity': '{:.2f}'}), height=300)
                    historical_month_agg = aggregate_week_to_month(historical_data, period_col)
                    st.subheader("Đồ thị tổng hợp theo tháng")
                    plot_historical_and_forecast(historical_month_agg, df_month_agg, 'Month')

