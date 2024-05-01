import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Stock Data Fetching", layout="wide")

# Sidebar File Uploader
with st.sidebar:
    upload_file = st.file_uploader("Choose a CSV file with stock symbols and dates", type=["csv"])
    if not upload_file:
        st.warning("Please upload a file to proceed.")
        st.stop()

# Load the data
data = pd.read_csv(upload_file)

# Convert 'date' to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Find unique stock symbols and their date ranges
stock_symbols = data['stock'].unique()

# Initialize a dictionary to hold stock data
stock_data = {}

# Fetch stock data for each symbol
for symbol in stock_symbols:
    # Get the earliest and latest date for the current stock symbol
    date_range = data[data['stock'] == symbol]['date']
    start_date = date_range.min()  # Earliest date
    end_date = date_range.max()  # Latest date
    
    # Fetch stock data using yfinance
    stock_df = yf.download(symbol, start=start_date, end=end_date)
    
    # Store the data in the dictionary
    stock_data[symbol] = stock_df

# Display fetched stock data
st.write("Fetched Stock Data:")
for symbol in stock_data:
    st.write(f"Stock Data for {symbol}:")
    st.dataframe(stock_data[symbol].head())  # Display the first few rows
