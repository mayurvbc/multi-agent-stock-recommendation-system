import streamlit as st
from analyze import analyze_stock, backtest_stock, report_backtest
from config import tickers_pool
from utils import format_debate, format_entry_exit
import pandas as pd
import matplotlib.pyplot as plt

st.title("Multi-Agent Stock Recommendation System")

if st.button("Generate Recommendations"):
    top_tickers = tickers_pool[:5]
    top_results = [analyze_stock(t) for t in top_tickers]
    df_top = pd.DataFrame(top_results)

    df_top['Debate Transcript'] = df_top['Debate Transcript'].apply(format_debate)
    df_top['Entry/Exit Levels'] = df_top['Entry/Exit Levels'].apply(format_entry_exit)

    st.subheader("Top 5 High-Volume Stocks")
    st.dataframe(df_top[['Ticker', 'Current Price', 'Target Price', 'Suggested Timeline',
                         'Recommendation', 'Confidence (%)', 'Risk Level', 'Latest Data']])

    st.markdown("---")
    st.subheader("Detailed Analysis")
    for i, row in df_top.iterrows():
        with st.expander(f"{row['Ticker']} - {row['Recommendation']} ({row['Confidence (%)']}%, {row['Risk Level']})"):
            st.text(f"Current Price: {row['Current Price']}")
            st.text(f"Target Price: {row['Target Price']} ({row['Suggested Timeline']})")
            st.text("Debate Transcript:\n" + row['Debate Transcript'])
            st.text("Entry/Exit Levels:\n" + str(row['Entry/Exit Levels']))
            st.text(f"Latest Data Fetched: {row['Latest Data']}")

st.markdown("---")
st.header("Backtesting Module")

ticker = st.selectbox("Select Stock for Backtest:", options=tickers_pool)
hold_period = st.slider("Holding Period (days):", min_value=1, max_value=30, value=5)

if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        df_results = backtest_stock(ticker, period="3y", hold_period=hold_period)
        report = report_backtest(df_results)

        st.success("Backtest Complete!")
        st.subheader("Summary Report")
        st.write(report)

        st.subheader("Detailed Results")
        st.dataframe(df_results)

        df_results['Date'] = pd.to_datetime(df_results['Date'])
        df_results.set_index('Date', inplace=True)
        success_rate = df_results['Success'].rolling(window=30).mean() * 100

        plt.figure(figsize=(10, 5))
        plt.plot(success_rate, label="Rolling 30-day Success Rate (%)")
        plt.xlabel("Date")
        plt.ylabel("Success Rate (%)")
        plt.title(f"Backtest Rolling Success Rate: {ticker}")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        csv = df_results.to_csv().encode()
        st.download_button(
            label="Download Backtest Results as CSV",
            data=csv,
            file_name=f"backtest_{ticker}.csv",
            mime="text/csv"
        )
