import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

# Download stock data
start = '2012-01-01'
end = '2024-04-20'
stock = 'GOOG'
data = yf.download(stock, start, end)
data.reset_index(inplace=True)

# Data preprocessing
data.dropna(inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Close']])

# Prepare data for training
window_size = 100  # Adjust window size as needed
future_days = 365  # Number of days to predict

x_train, y_train = [], []

for i in range(window_size, len(data_scaled)):
    x_train.append(data_scaled[i-window_size:i, 0])
    y_train.append(data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=150, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=160, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=280, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=320))
model.add(Dropout(0.5))
model.add(Dense(units=1))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100, batch_size=32)

# Save the model
model.save('StockPredictionsModel.h5')

# Prepare test data for prediction
data_test = data[-window_size:]  
data_test_scaled = scaler.transform(data_test[['Close']])
x_test = np.array([data_test_scaled])

# Make predictions on test data
predicted_prices = model.predict(x_test) 

# Reshape predicted prices if needed
predicted_prices = np.squeeze(predicted_prices)  
predicted_prices = predicted_prices.reshape(-1, 1)  

# Inverse transform the predicted prices
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plotting predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Historical Prices')
plt.plot(range(len(data), len(data) + len(predicted_prices)), predicted_prices, label='Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()