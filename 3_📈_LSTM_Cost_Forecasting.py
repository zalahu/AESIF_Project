import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="AESIF | Forecast", layout="wide")
st.title("📈 Time-Series Forecasting (LSTM)")

st.markdown("Leverage Long Short-Term Memory (LSTM) neural networks to forecast future capital expenditures or resource utilization based on sequential historical data.")

uploaded_file = st.file_uploader("Upload Time-Series Data (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    date_col = st.selectbox("Select Date/Time Column", df.columns)
    val_col = st.selectbox("Select Value Column to Forecast", df.select_dtypes(include=[np.number]).columns)
    
    look_back = st.slider("Look-back Window (Time Steps)", 5, 60, 10)
    forecast_steps = st.slider("Periods to Forecast Ahead", 5, 30, 12)
    
    if st.button("Train LSTM & Generate Forecast"):
        with st.spinner("Building and training neural network (this may take a moment)..."):
            # Data Preparation
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            data = df[val_col].values.reshape(-1, 1)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(len(scaled_data) - look_back):
                X.append(scaled_data[i:(i + look_back), 0])
                y.append(scaled_data[i + look_back, 0])
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # LSTM Architecture
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Fast training for live demo purposes
            progress_bar = st.progress(0)
            epochs = 10
            for epoch in range(epochs):
                model.fit(X, y, epochs=1, batch_size=32, verbose=0)
                progress_bar.progress((epoch + 1) / epochs)
                
            # Forecasting Future Steps
            last_sequence = scaled_data[-look_back:]
            current_batch = last_sequence.reshape((1, look_back, 1))
            forecast = []
            
            for _ in range(forecast_steps):
                current_pred = model.predict(current_batch)[0]
                forecast.append(current_pred)
                current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
                
            forecast = scaler.inverse_transform(forecast)
            
            # Generate Future Dates
            last_date = df[date_col].iloc[-1]
            # Assuming monthly frequency for demo; adapt as needed
            future_dates = pd.date_range(last_date, periods=forecast_steps+1, freq='M')[1:]
            
            # Plotting with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[date_col], y=df[val_col], mode='lines', name='Historical Data', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=future_dates, y=forecast.flatten(), mode='lines', name='LSTM Forecast', line=dict(color='red', dash='dash')))
            
            fig.update_layout(title="LSTM Predictive Forecast", xaxis_title="Date", yaxis_title="Value", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.success("Forecast generated successfully.")