import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

def display_features_table(df, period_col):
    st.dataframe(df)

def display_metrics_table(metrics_df):
    st.dataframe(metrics_df)

def display_stats(metrics_df):
    mean_mae = metrics_df['MAE_%'].mean()
    std_mae = metrics_df['MAE_%'].std()
    st.write(f"Mean MAE%: {mean_mae:.2f}%")
    st.write(f"Std MAE%: {std_mae:.2f}%")

def plot_forecast_vs_actual(test_plot_data, iteration, period_col):
    # Lấy dữ liệu từ test_plot_data
    periods = test_plot_data['Period'].values[0]  # Lấy danh sách giá trị Period
    actual = test_plot_data['Thực tế'].values[0]  # Lấy mảng giá trị Thực tế
    forecast = test_plot_data['Dự báo'].values[0]  # Lấy mảng giá trị Dự báo

    # Tạo đồ thị
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=actual, mode='lines+markers', name='Thực tế', line=dict(color='#00CC96')))
    fig.add_trace(go.Scatter(x=periods, y=forecast, mode='lines+markers', name='Dự báo', line=dict(color='#EF553B')))
    fig.update_layout(
        title=f'Dự báo vs Thực tế - Iteration {iteration}',
        xaxis_title=period_col,  # Sử dụng period_col làm nhãn trục x
        yaxis_title='Quantity',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark'
    )

    # Hiển thị bảng 24 tuần
    table_data = pd.DataFrame({
        period_col: periods,  # Sử dụng period_col cho cột thời gian
        'Thực tế': actual,
        'Dự báo': forecast
    }).round(2)  # Làm tròn 2 chữ số thập phân

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(table_data)

def plot_error_histogram(errors, iteration, is_nn=False):
    fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=20, marker_color='#636EFA' if not is_nn else '#AB63FA')])
    fig.update_layout(
        title=f'Histogram Sai Số - Iteration {iteration}',
        xaxis_title='Sai Số',
        yaxis_title='Số Lượng',
        bargap=0.1,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_and_forecast(historical_data, forecast_data, period_col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data[period_col], y=historical_data['Quantity'], mode='lines+markers', name='Thực tế', line=dict(color='#00CC96')))
    fig.add_trace(go.Scatter(x=forecast_data[period_col], y=forecast_data['Quantity'], mode='lines+markers', name='Dự báo', line=dict(color='#EF553B')))
    
    # Add bridge trace nối điểm cuối thực tế với điểm đầu dự báo
    if historical_data is not None and forecast_data is not None:
        if not historical_data.empty and not forecast_data.empty:
            last_hist_date = historical_data[period_col].iloc[-1]
            last_hist_value = historical_data['Quantity'].iloc[-1]
            first_forecast_date = forecast_data[period_col].iloc[0]
            first_forecast_value = forecast_data['Quantity'].iloc[0]

            fig.add_trace(go.Scatter(
                x=[last_hist_date, first_forecast_date],
                y=[last_hist_value, first_forecast_value],
                mode='lines',
                line=dict(color='#EF553B'),
                showlegend=False
            ))
    
    fig.update_layout(
        title='Chuỗi Thực Tế và Dự Báo',
        xaxis_title=period_col,
        yaxis_title='Quantity',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

def display_actual_vs_forecast_table(historical_data, forecast_data, period_col):
    combined_data = pd.concat([historical_data, forecast_data])
    st.dataframe(combined_data.style.format({'Quantity': '{:.2f}'}), height=300)