import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Stock Forecasting Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.write("Preview of data:", df.head())

    # Simple line chart
    st.line_chart(df.set_index("Date")["Close"])
