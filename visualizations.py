import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def display_features_table(df, period_col):
    available_cols = [col for col in [period_col, 'y', 'y_log', 'Quantity_don_hang', 'Quantity_sua_chua', 
                                     'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_12', 
                                     'non_zero', 'rolling_mean', 'rolling_std', 'month_of_year', 'week_of_year', 
                                     'seasonal_index', 'peak_period', 'OrderFormName', 'Nhóm ĐL'] 
                     if col in df.columns]
    st.dataframe(df[available_cols].style.format({
        'y': '{:.2f}', 'y_log': '{:.2f}', 'Quantity_don_hang': '{:.2f}', 'Quantity_sua_chua': '{:.2f}',
        'lag_1': '{:.2f}', 'lag_2': '{:.2f}', 'lag_3': '{:.2f}', 'lag_4': '{:.2f}', 
        'lag_12': '{:.2f}', 'rolling_mean': '{:.2f}', 'rolling_std': '{:.2f}', 
        'seasonal_index': '{:.2f}'
    }), height=300)

def display_metrics_table(top_10_metrics):
    top_10_metrics['Test_Start'] = top_10_metrics['Test_Start'].dt.strftime('%Y-%m-%d')
    top_10_metrics['Test_End'] = top_10_metrics['Test_End'].dt.strftime('%Y-%m-%d')
    st.dataframe(top_10_metrics.style.format({
        'MAE': '{:.2f}',
        'MAE_%': '{:.2f}%',
        'RMSE': '{:.2f}',
        'RMSE_%': '{:.2f}%',
        'MAPE_%': '{:.2f}%'
    }), height=300)

def display_stats(top_10_metrics):
    st.write(f"Mean MAE%: {top_10_metrics['MAE_%'].mean():.2f}%")
    st.write(f"Std MAE%: {top_10_metrics['MAE_%'].std():.2f}%")
    st.write(f"Mean RMSE%: {top_10_metrics['RMSE_%'].mean():.2f}%")
    st.write(f"Std RMSE%: {top_10_metrics['RMSE_%'].std():.2f}%")
    st.write(f"Mean MAPE%: {top_10_metrics['MAPE_%'].mean():.2f}%")
    st.write(f"Std MAPE%: {top_10_metrics['MAPE_%'].std():.2f}%")

def plot_forecast_vs_actual(test_plot_data, selected_iteration, period_col):
    title = f"Dự báo trên tập test (Iteration {selected_iteration})"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_plot_data['Period'].values[0], y=test_plot_data['Thực tế'].values[0], 
                             mode='lines+markers', name='Thực tế', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_plot_data['Period'].values[0], y=test_plot_data['Dự báo'].values[0], 
                             mode='lines+markers', name='Dự báo', line=dict(color='orange', dash='dash')))
    fig.update_layout(
        title=title,
        xaxis_title=period_col,
        yaxis_title="Số lượng xuất kho",
        template="plotly_white",
        showlegend=True,
        xaxis=dict(tickangle=45, tickformat='%Y-%m-%d')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Thêm bảng hiển thị dữ liệu của biểu đồ với 2 cột Thực tế và Dự báo
    st.subheader(f"Dữ liệu chuỗi thực tế và dự báo trên tập test (Iteration {selected_iteration})")
    periods = test_plot_data['Period'].values[0]
    actual = test_plot_data['Thực tế'].values[0]
    predicted = test_plot_data['Dự báo'].values[0]
    
    combined = pd.DataFrame({
        period_col: periods,
        'Thực tế': actual,
        'Dự báo': predicted
    })
    combined[period_col] = pd.to_datetime(combined[period_col]).dt.strftime('%Y-%m-%d')
    st.dataframe(combined.style.format({'Thực tế': '{:.2f}', 'Dự báo': '{:.2f}'}), height=600)

def plot_error_histogram(errors, selected_iteration, is_nn=False):
    title = f"Phân phối sai số (Iteration {selected_iteration})"
    fig = px.histogram(x=errors, nbins=20, title=title)
    fig.update_layout(
        xaxis_title="Sai số (Thực tế - Dự báo)",
        yaxis_title="Tần suất",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_and_forecast(historical_data, df_future, period_col):
    title = f"Dự báo nhu cầu xuất kho 6 {period_col.lower()} tiếp theo"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data[period_col], y=historical_data['Quantity'], 
                             mode='lines+markers', name='Thực tế', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_future[period_col], y=df_future['Quantity'], 
                             mode='lines+markers', name='Dự báo', line=dict(color='orange', dash='dash')))
    fig.update_layout(
        title=title,
        xaxis_title=period_col,
        yaxis_title="Số lượng xuất kho",
        template="plotly_white",
        showlegend=True,
        xaxis=dict(tickangle=45, tickformat='%Y-%m-%d')
    )
    st.plotly_chart(fig, use_container_width=True)



def display_test_actual_vs_forecast_table(test_plot_data, selected_iteration, period_col):
    st.subheader(f"Bảng chuỗi thực tế và dự báo trên tập test (Iteration {selected_iteration})")
    periods = test_plot_data['Period'].values[0]
    actual = test_plot_data['Thực tế'].values[0]
    predicted = test_plot_data['Dự báo'].values[0]
    
    actual_df = pd.DataFrame({
        period_col: periods,
        'Quantity': actual,
        'Type': 'Thực tế'
    })
    predicted_df = pd.DataFrame({
        period_col: periods,
        'Quantity': predicted,
        'Type': 'Dự báo'
    })
    combined = pd.concat([actual_df, predicted_df], ignore_index=True)
    combined[period_col] = pd.to_datetime(combined[period_col]).dt.strftime('%Y-%m-%d')
    st.dataframe(combined.style.format({'Quantity': '{:.2f}'}), height=600)