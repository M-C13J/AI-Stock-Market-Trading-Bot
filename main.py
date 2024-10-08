import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import time

# Load the trained model
model = None
try:
    model = load_model('StockPredictionsModel.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

def main():
    st.sidebar.header('Trading Parameters')
    stock = st.sidebar.text_input('Enter Stock Symbol', 'TSLA')
    start = '2022-01-01'
    end = '2024-04-15'
    data = yf.download(stock, start, end)
    st.subheader('Stock Data')
    st.write(data)

    if not data.empty and model is not None:
        # Data preprocessing
        data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_train)  # Fit scaler on training data
        data_train_scaled = scaler.transform(data_train)
        data_test_scaled = scaler.transform(data_test)

        # Prepare data for predictions
        pas_100_days = data_train_scaled[-100:]
        data_test_scaled = np.concatenate([pas_100_days, data_test_scaled], axis=0)

        x = []
        y = []
        for i in range(100, len(data_test_scaled)):
            x.append(data_test_scaled[i-100:i])
            y.append(data_test_scaled[i, 0])
        x, y = np.array(x), np.array(y)

        # Make predictions on test data
        try:
            predicted_prices = model.predict(x)
            predicted_prices = scaler.inverse_transform(predicted_prices)  # Inverse transform predictions
            y = scaler.inverse_transform(y.reshape(-1, 1))  # Inverse transform actual values
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            return

        # Plot actual vs predicted prices
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.plot(data_test.index, y, 'g', label='Actual Price')
        ax4.plot(data_test.index, predicted_prices, 'r', label='Predicted Price')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price')
        ax4.set_title(f'Original Price vs Predicted Price for {stock}')
        ax4.legend()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator())
        fig4.autofmt_xdate()
        st.pyplot(fig4)

        # Prediction for future days
        x_future = data_test_scaled[-100:].reshape((1, 100, 1))
        future_days = 365  # Number of future days to predict
        predicted_future_prices = []

        for _ in range(future_days):
            pred = model.predict(x_future)
            noise = np.random.normal(0, 0.02)  # Increased noise level for more variation
            pred_with_noise = pred + noise

            # Check if prediction is too extreme, reset to more realistic value if necessary
            if pred_with_noise[0, 0] > 1.5 or pred_with_noise[0, 0] < 0:  # Adjust thresholds as needed
                pred_with_noise[0, 0] = pred[0, 0]  # Reset to original prediction without noise

            predicted_future_prices.append(pred_with_noise[0, 0])
            x_future = np.append(x_future[:, 1:, :], pred_with_noise.reshape(1, 1, 1), axis=1)

        # Inverse transform predicted future prices
        predicted_future_prices = scaler.inverse_transform(np.array(predicted_future_prices).reshape(-1, 1))

        # Create a combined array of actual and predicted prices for smooth plotting
        combined_prices = np.concatenate([data_test.Close.values, predicted_future_prices.flatten()])

        # Create a date range for the future predictions
        future_dates = pd.date_range(data_test.index[-1], periods=future_days + 1)[1:]

        # Plot future predictions
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        ax5.plot(data_test.index, data_test.Close.values, label='Actual Price')
        ax5.plot(future_dates, predicted_future_prices, label='Predicted Price')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Price')
        ax5.set_title(f'Future Price Prediction for {stock}')
        ax5.legend()
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax5.xaxis.set_major_locator(mdates.MonthLocator())
        fig5.autofmt_xdate()
        st.pyplot(fig5)

        # Set trading parameters
        stop_loss_percentage = st.sidebar.slider('Stop Loss Percentage', 0.1, 10.0, 2.0)
        take_profit_percentage = st.sidebar.slider('Take Profit Percentage', 0.1, 10.0, 2.0)
        portfolio_value = st.sidebar.number_input('Initial Portfolio Value', min_value=1000, value=10000)
        trade_percentage = st.sidebar.slider('Percentage of Portfolio to Invest per Trade', 1, 10, 5)

        start_trading = st.sidebar.button('Start Trading')

        if start_trading:
            # Portfolio management
            cash_balance = portfolio_value
            shares_held = 0  # Shares currently held

            portfolio_values = []  # List to store portfolio values for plotting
            buy_signals = []  # List to store buy signals
            sell_signals = []  # List to store sell signals

            # Placeholder for updating the plots dynamically
            price_plot_placeholder = st.empty()
            portfolio_plot_placeholder = st.empty()

            stop_loss_level = None  # Initialize stop loss level
            take_profit_level = None  # Initialize take profit level

            for i in range(len(data_test)):
                current_price = data_test.Close.iloc[i]
                predicted_price = predicted_prices[i]

                # Check for buy signal
                if predicted_price > current_price and shares_held == 0:
                    amount_to_invest = cash_balance * (trade_percentage / 100)
                    shares_held = int(amount_to_invest / current_price)
                    cash_balance -= shares_held * current_price
                    buy_signals.append((data_test.index[i], current_price))

                    # Set stop loss and take profit levels
                    stop_loss_level = current_price * (1 - stop_loss_percentage / 100)
                    take_profit_level = current_price * (1 + take_profit_percentage / 100)

                # Check for sell signal
                elif shares_held > 0 and (current_price <= stop_loss_level or current_price >= take_profit_level):
                    cash_balance += shares_held * current_price
                    shares_held = 0
                    sell_signals.append((data_test.index[i], current_price))

                    # Reset stop loss and take profit levels after selling
                    stop_loss_level = None
                    take_profit_level = None

                # Update portfolio value
                portfolio_value = cash_balance + shares_held * current_price
                portfolio_values.append(portfolio_value)

                # Update price plot with buy and sell signals
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data_test.index[:i+1], data_test.Close.values[:i+1], label='Actual Price')
                ax.plot(data_test.index[:i+1], predicted_prices[:i+1], label='Predicted Price', linestyle='--')
                for buy_signal in buy_signals:
                    ax.plot(buy_signal[0], buy_signal[1], marker='^', color='g')
                for sell_signal in sell_signals:
                    ax.plot(sell_signal[0], sell_signal[1], marker='v', color='r')
                if shares_held > 0:
                    ax.axhline(y=stop_loss_level, color='r', linestyle='dashed', alpha=0.5)
                    ax.axhline(y=take_profit_level, color='g', linestyle='dashed', alpha=0.5)
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.set_title(f'Trading Simulation for {stock}')
                ax.legend()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                fig.autofmt_xdate()
                price_plot_placeholder.pyplot(fig)

                # Update portfolio value plot
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(data_test.index[:i+1], portfolio_values, label='Portfolio Value')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Portfolio Value')
                ax2.set_title('Portfolio Value Over Time')
                ax2.legend()
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax2.xaxis.set_major_locator(mdates.MonthLocator())
                fig2.autofmt_xdate()
                portfolio_plot_placeholder.pyplot(fig2)

                # Simulate live trading with a small delay
                time.sleep(0.1)
    else:
        st.warning("Model could not be loaded or no data available for the specified stock.")

if __name__ == "__main__":
    main()
