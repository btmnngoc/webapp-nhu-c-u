import pandas as pd
import numpy as np
import streamlit as st

def tai_va_xu_ly_du_lieu(file_path):
    try:
        don_hang = pd.read_excel(file_path, sheet_name='Danh mục đơn hàng giả định')
        sua_chua = pd.read_excel(file_path, sheet_name='Phiếu sửa chữa')
        
        if not pd.api.types.is_datetime64_any_dtype(don_hang['DocDate']):
            don_hang['DocDate'] = pd.to_datetime(don_hang['DocDate'], origin='1899-12-30', unit='D', errors='coerce')
        
        if pd.api.types.is_numeric_dtype(sua_chua['Ngày']):
            sua_chua['Date'] = pd.to_datetime(sua_chua['Ngày'], origin='1899-12-30', unit='D', errors='coerce')
        else:
            try:
                sua_chua['Date'] = pd.to_datetime(sua_chua['Ngày'], errors='coerce', dayfirst=True)
            except Exception as e:
                st.error(f"Không thể chuyển đổi định dạng ngày: {str(e)}")
                return None, None
        
        return don_hang, sua_chua
    except Exception as e:
        st.error(f"Lỗi load dữ liệu: {str(e)}")
        return None, None

def tong_hop_nhu_cau(don_hang, sua_chua, ma_san_pham, granularity='M'):
    don_hang_sp = don_hang[don_hang['ItemCode'] == ma_san_pham][['DocDate', 'Quantity', 'OrderFormName', 'Nhóm ĐL']].copy()
    sua_chua_sp = sua_chua[sua_chua['Mã PT'] == ma_san_pham][['Date', 'Slg']].copy()

    if don_hang_sp.empty and sua_chua_sp.empty:
        st.error(f"Không có dữ liệu cho mã {ma_san_pham}")
        return None, None

    if granularity == 'M':
        period_col = 'Month'
        freq = 'MS'
        lags = [1, 2, 3, 4, 12]
        seasonal_div = 12
        date_col = 'month_of_year'
        max_date = pd.Timestamp('2025-09-01')
    else:  # 'W'
        period_col = 'Week'
        freq = 'W-MON'
        lags = [1, 2, 3, 4, 12]
        seasonal_div = 52
        date_col = 'week_of_year'
        max_date = pd.Timestamp('2025-08-25')

    don_hang_sp[period_col] = don_hang_sp['DocDate'].dt.to_period(granularity).apply(lambda r: r.start_time)
    sua_chua_sp[period_col] = sua_chua_sp['Date'].dt.to_period(granularity).apply(lambda r: r.start_time)

    don_hang_agg = don_hang_sp.groupby(period_col).agg({
        'Quantity': 'sum',
        'OrderFormName': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'Nhóm ĐL': lambda x: x.mode()[0] if not x.empty else 'Unknown'
    }).reset_index()
    sua_chua_agg = sua_chua_sp.groupby(period_col)['Slg'].sum().reset_index()
    sua_chua_agg.rename(columns={'Slg': 'Quantity'}, inplace=True)

    nhu_cau_agg = pd.merge(don_hang_agg, sua_chua_agg, on=period_col, how='outer', suffixes=('_don_hang', '_sua_chua'))
    nhu_cau_agg['y'] = pd.to_numeric(
        nhu_cau_agg['Quantity_don_hang'].fillna(0) + nhu_cau_agg['Quantity_sua_chua'].fillna(0),
        errors='coerce'
    ).fillna(0)

    min_date = nhu_cau_agg[period_col].min()
    all_periods = pd.date_range(start=min_date, end=max_date, freq=freq).to_frame(index=False, name=period_col)
    nhu_cau_agg = pd.merge(all_periods, nhu_cau_agg, on=period_col, how='left')
    nhu_cau_agg['y'] = pd.to_numeric(nhu_cau_agg['y'], errors='coerce').fillna(0)
    nhu_cau_agg['Quantity_don_hang'] = nhu_cau_agg['Quantity_don_hang'].fillna(0)
    nhu_cau_agg['Quantity_sua_chua'] = nhu_cau_agg['Quantity_sua_chua'].fillna(0)
    nhu_cau_agg['OrderFormName'] = nhu_cau_agg['OrderFormName'].fillna('Unknown')
    nhu_cau_agg['Nhóm ĐL'] = nhu_cau_agg['Nhóm ĐL'].fillna('Unknown')
    
    for lag in lags:
        nhu_cau_agg[f'lag_{lag}'] = nhu_cau_agg['y'].shift(lag).fillna(0).astype(float)
    nhu_cau_agg['non_zero'] = (nhu_cau_agg['y'] > 0).astype(int)
    nhu_cau_agg['rolling_mean'] = nhu_cau_agg['y'].rolling(window=3).mean().fillna(0).astype(float)
    nhu_cau_agg['rolling_std'] = nhu_cau_agg['y'].rolling(window=3).std().fillna(0).astype(float)
    
    if granularity == 'M':
        nhu_cau_agg['month_of_year'] = nhu_cau_agg[period_col].dt.month
        nhu_cau_agg['seasonal_index'] = np.sin(2 * np.pi * nhu_cau_agg['month_of_year'] / seasonal_div)
        nhu_cau_agg['peak_period'] = nhu_cau_agg[period_col].dt.month.isin([1, 2, 11, 12]).astype(int)
    else:
        nhu_cau_agg['week_of_year'] = nhu_cau_agg[period_col].dt.isocalendar().week
        nhu_cau_agg['seasonal_index'] = np.sin(2 * np.pi * nhu_cau_agg['week_of_year'] / seasonal_div)
        nhu_cau_agg['peak_period'] = nhu_cau_agg['week_of_year'].isin([1, 52]).astype(int)
    
    nhu_cau_agg['y_log'] = np.log1p(nhu_cau_agg['y'] + 0.1)
    nhu_cau_agg['ItemCode'] = ma_san_pham
    
    return nhu_cau_agg, period_col