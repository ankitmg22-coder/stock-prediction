import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf 
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

start = '2010-01-01'
end = '2026-03-26'

st.title('📈 Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# ---------------- DATA ----------------
st.subheader('Data from 2010-2026')
st.write(df.describe())

# ---------------- CHARTS ----------------
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# ---------------- SPLIT ----------------
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# ---------------- TRAIN MODEL ----------------
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i])

x_train, y_train = np.array(x_train), np.array(y_train)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 🔥 fast training
model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=0)

# ---------------- TEST ----------------
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test)

# ---------------- LIVE PRICE ----------------
df_live = yf.download(user_input, period="5y")
current_price = float(df_live['Close'].iloc[-1])

# ---------------- GRAPH ----------------
st.subheader('Predictions vs Original + Live')

fig2 = plt.figure(figsize=(12,6))

plt.plot(y_test, label='Original', linewidth=2)
plt.plot(y_predicted, label='Predicted', linewidth=2)

plt.axhline(y=current_price, linestyle='--', label='Live Price')

plt.grid(alpha=0.3)
plt.legend()
st.pyplot(fig2)

df_live = yf.download(user_input, period="5y")

if not df_live.empty:
    current_price = float(df_live['Close'].iloc[-1])

    st.subheader("📍 Current Live Price")
    st.success(f"₹ {current_price:.2f}")
else:
    st.error("Live price not available ❌")


st.sidebar.title("FILTER 📈 ")

ticker = st.sidebar.text_input("Enter Stock", "AAPL")
days = st.sidebar.slider("Future Days", 10, 100, 10)



st.line_chart(df['Close'])



previous_price = float(df_live['Close'].iloc[-2])

change = current_price - previous_price
percent = (change / previous_price) * 100




st.set_page_config(layout="wide")

col1, col2 = st.columns(2)

with col1:
    st.metric("Price", f"${current_price:.2f}")

with col2:
    st.metric("Change", f"{change:.2f}", f"{percent:.2f}%")




















# ---------------- FUTURE PREDICTION ----------------
st.subheader('🔮 Future Predictions')

data = df[['Close']]
data_scaled = scaler.fit_transform(data)

last_data = data_scaled[-100:]
time_step = 100

# ---------- NEXT 10 DAYS ----------
future_10 = []
current_input = last_data.copy()

for i in range(10):
    current_input_reshaped = current_input.reshape(1, time_step, 1)
    pred = model.predict(current_input_reshaped, verbose=0)
    
    future_10.append(pred[0][0])
    current_input = np.vstack((current_input[1:], pred))

future_10 = np.array(future_10).reshape(-1,1)
future_10 = scaler.inverse_transform(future_10)

from datetime import timedelta
from datetime import datetime, timedelta

last_date = datetime.now()
dates_10 = pd.bdate_range(start=datetime.now(), periods=10)
df_10 = pd.DataFrame({
    "Date": dates_10,
    "Predicted Price": future_10.flatten()
})

st.subheader("📅 Next 10 Days Prediction")
st.write(df_10)

# ---------- NEXT 100 DAYS ----------
future_100 = []
current_input = last_data.copy()

for i in range(100):
    current_input_reshaped = current_input.reshape(1, time_step, 1)
    pred = model.predict(current_input_reshaped, verbose=0)
    
    future_100.append(pred[0][0])
    current_input = np.vstack((current_input[1:], pred))

future_100 = np.array(future_100).reshape(-1,1)
future_100 = scaler.inverse_transform(future_100)

st.subheader("📈 Next 100 Days Prediction")

fig3 = plt.figure(figsize=(12,6))
plt.plot(future_100, 'g', label='Future')
plt.legend()

st.pyplot(fig3)