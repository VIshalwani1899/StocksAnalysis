import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Function to get stock data (same as backend)
def get_stock_data(stock_ticker):
    yfin = yf.Ticker(stock_ticker)
    hist = yfin.history(period="max")
    hist = hist[['Close']]
    hist.reset_index(inplace=True)
    hist.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    hist['ds'] = pd.to_datetime(hist['ds']).dt.tz_localize(None)  # Remove timezone
    return hist

# Function to predict future price
def predict_future_price(hist, forecast_days):
    m = Prophet(weekly_seasonality=False)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(hist)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    return forecast, m

# Function to visualize stock data and forecast
def visualize_stock_data_forecast(hist, forecast, m):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # Plot historical data and forecasted price
    axs[0].plot(hist['ds'], hist['y'], color='blue', label='Historical Price')
    axs[0].plot(forecast['ds'], forecast['yhat'], color='red', linestyle='--', label='Forecasted Price')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].set_title('Stock Price Forecast')
    axs[0].legend()

    # Try using plot with kind='seasonality' if supported (check Prophet documentation)
    try:
        m.plot(forecast, kind='seasonality')
        axs[1].set_title('Seasonality')
    except TypeError:  # 'kind' argument might not be supported (fallback)
        m.plot_components(forecast)
        axs[1].set_title('Forecast Components')

    plt.tight_layout()

    # Convert the Matplotlib figure to a Streamlit image for display (remove caption argument)
    st.pyplot(fig)  # Removed caption argument

# Upgrade Pillow Library (optional, but recommended)
try:
  # Check if Pillow is installed
  import PIL
except ModuleNotFoundError:
  # Inform user about missing library (assuming user has necessary permissions)
  st.warning('Pillow library not found. Consider installing it for potential image functionalities: pip install Pillow')

def main():
    # Create textboxes and dropdown for user input
    stock_ticker = st.text_input("Enter stock ticker symbol (e.g., INFY.NS): ")

    # Ensure forecast_days is an integer (avoid potential string conversion)
    selected_day_index = st.selectbox("Select the number of forecast days:", range(30, 366, 30))
    forecast_days = selected_day_index * 30  # Calculate actual forecast days

    if st.button("Predict"):  # Button click triggers the prediction
        try:
            hist = get_stock_data(stock_ticker)
            forecast, m = predict_future_price(hist, forecast_days)
            visualize_stock_data_forecast(hist, forecast, m)

        except Exception as e:
            st.error(f"Error processing {stock_ticker}: {e}")

if __name__ == "__main__":
    main()
