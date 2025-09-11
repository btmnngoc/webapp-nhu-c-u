import pandas as pd
import numpy as np
import streamlit as st

def tai_va_xu_ly_du_lieu(file_path):
    try:
        # Đọc đúng sheet "Danh mục xuất kho"
        don_hang = pd.read_excel(file_path, sheet_name='Danh mục xuất hàng')

        # Chuẩn hóa cột ngày
        if not pd.api.types.is_datetime64_any_dtype(don_hang['DocDate']):
            try:
                don_hang['DocDate'] = pd.to_datetime(don_hang['DocDate'], errors='coerce', dayfirst=True)
            except Exception as e:
                st.error(f"Không thể chuyển đổi định dạng ngày: {str(e)}")
                return None

        return don_hang
    except Exception as e:
        st.error(f"Lỗi load dữ liệu: {str(e)}")
        return None

def tong_hop_nhu_cau(don_hang, ma_san_pham, granularity='M'):
    # Lọc theo mã sản phẩm
    don_hang_sp = don_hang[don_hang['ItemCode'] == ma_san_pham][['DocDate', 'Quantity']].copy()

    if don_hang_sp.empty:
        st.error(f"Không có dữ liệu cho mã {ma_san_pham}")
        return None, None

    # Định nghĩa chu kỳ
    if granularity == 'M':
        period_col = 'Month'
        freq = 'MS'
        lags = [1, 2, 3, 4, 12]
        seasonal_div = 12
    else:  # 'W'
        period_col = 'Week'
        freq = 'W-MON'
        lags = [1, 2, 3, 4, 12]
        seasonal_div = 52

    # Chuẩn hóa thời gian
    don_hang_sp[period_col] = don_hang_sp['DocDate'].dt.to_period(granularity).apply(lambda r: r.start_time)

    # Gộp dữ liệu theo kỳ
    nhu_cau_agg = don_hang_sp.groupby(period_col)['Quantity'].sum().reset_index()
    nhu_cau_agg.rename(columns={'Quantity': 'y'}, inplace=True)

    # Điền thiếu kỳ
    min_date, max_date = nhu_cau_agg[period_col].min(), nhu_cau_agg[period_col].max()
    all_periods = pd.date_range(start=min_date, end=max_date, freq=freq).to_frame(index=False, name=period_col)
    nhu_cau_agg = pd.merge(all_periods, nhu_cau_agg, on=period_col, how='left')
    nhu_cau_agg['y'] = nhu_cau_agg['y'].fillna(0)

    # Lag features
    for lag in lags:
        nhu_cau_agg[f'lag_{lag}'] = nhu_cau_agg['y'].shift(lag).fillna(0).astype(float)

    # Các feature bổ sung
    nhu_cau_agg['non_zero'] = (nhu_cau_agg['y'] > 0).astype(int)
    nhu_cau_agg['rolling_mean'] = nhu_cau_agg['y'].rolling(window=3).mean().fillna(0).astype(float)
    nhu_cau_agg['rolling_std'] = nhu_cau_agg['y'].rolling(window=3).std().fillna(0).astype(float)

    # Seasonality features
    if granularity == 'M':
        nhu_cau_agg['month_of_year'] = nhu_cau_agg[period_col].dt.month
        nhu_cau_agg['seasonal_index'] = np.sin(2 * np.pi * nhu_cau_agg['month_of_year'] / seasonal_div)
        nhu_cau_agg['peak_period'] = nhu_cau_agg[period_col].dt.month.isin([1, 2, 11, 12]).astype(int)
    else:
        nhu_cau_agg['week_of_year'] = nhu_cau_agg[period_col].dt.isocalendar().week
        nhu_cau_agg['seasonal_index'] = np.sin(2 * np.pi * nhu_cau_agg['week_of_year'] / seasonal_div)
        nhu_cau_agg['peak_period'] = nhu_cau_agg['week_of_year'].isin([1, 52]).astype(int)

    # Log transform
    nhu_cau_agg['y_log'] = np.log1p(nhu_cau_agg['y'])

    # Gắn mã sản phẩm
    nhu_cau_agg['ItemCode'] = ma_san_pham

    return nhu_cau_agg, period_col