import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('Stock Predictions Model.keras')

# Streamlit App Header
st.header('Stock Market Predictor')

# Input field for stock symbol (no default value)
stock = st.text_input('Enter Stock Symbol').strip().upper()

# Proceed only if a stock symbol is entered
if stock:
    start = '2012-01-01'
    end = '2024-12-31'

    # Fetch stock data with error handling
    try:
        data = yf.download(stock, start, end)

        # Check if data is empty
        if data.empty:
            st.error(f"Could not retrieve data for '{stock}'. Check the stock symbol and try again.")
        else:
            st.subheader('Stock Data')
            st.write(data.tail())  # Display last few rows

            # Splitting data into training and testing sets
            data_train = pd.DataFrame(data['Close'][0: int(len(data) * 0.80)])
            data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

            # MinMax Scaling
            scaler = MinMaxScaler(feature_range=(0,1))

            # Ensure there is enough data for testing
            if data_train.empty or data_test.empty:
                st.error("Not enough historical data to make predictions. Try another stock.")
            else:
                # Prepare test data
                past_100_days = data_train.tail(100)
                data_test = pd.concat([past_100_days, data_test], ignore_index=True)
                data_test_scaled = scaler.fit_transform(data_test)

                # Moving Averages
                st.subheader('Price vs MA50')
                ma_50 = data['Close'].rolling(50).mean()
                fig1 = plt.figure(figsize=(8,6))
                plt.plot(data['Close'], 'g', label='Closing Price')
                plt.plot(ma_50, 'r', label='MA50')
                plt.legend()
                st.pyplot(fig1)

                st.subheader('Price vs MA50 vs MA100')
                ma_100 = data['Close'].rolling(100).mean()
                fig2 = plt.figure(figsize=(8,6))
                plt.plot(data['Close'], 'g', label='Closing Price')
                plt.plot(ma_50, 'r', label='MA50')
                plt.plot(ma_100, 'b', label='MA100')
                plt.legend()
                st.pyplot(fig2)

                st.subheader('Price vs MA100 vs MA200')
                ma_200 = data['Close'].rolling(200).mean()
                fig3 = plt.figure(figsize=(8,6))
                plt.plot(data['Close'], 'g', label='Closing Price')
                plt.plot(ma_100, 'r', label='MA100')
                plt.plot(ma_200, 'b', label='MA200')
                plt.legend()
                st.pyplot(fig3)

                # Prepare input for model prediction
                x, y = [], []
                for i in range(100, data_test_scaled.shape[0]):
                    x.append(data_test_scaled[i-100:i])
                    y.append(data_test_scaled[i, 0])

                x, y = np.array(x), np.array(y)

                # Make predictions
                predictions = model.predict(x)

                # Rescale predictions
                scale_factor = 1 / scaler.scale_
                predictions = predictions * scale_factor
                y = y * scale_factor

                # Plot predictions
                st.subheader('Original Price vs Predicted Price')
                fig4 = plt.figure(figsize=(8,6))
                plt.plot(y, 'g', label='Original Price')
                plt.plot(predictions, 'r', label='Predicted Price')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig4)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter a stock symbol to proceed.")
