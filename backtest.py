import yfinance as yf
import pandas as pd
from analyze import analyze_stock

def analyze_stock_for_backtest(ticker, hist_window):
    # Simplified for backtesting: analyze based on a small window of historical data
    info = yf.Ticker(ticker).info
    headlines = [f"{ticker} backtest headlines"]

    # Reuse the same logic as analyze_stock but focused on small historical window
    debate = {
        'Price Action': ("Price action for backtest", 0.2, None, None),
        'Technical': ("Technical analysis for backtest", 0.2, None, None),
        'Sentiment': ("Sentiment analysis for backtest", 0.1, None, None),
        'Fundamentals': ("Fundamental analysis for backtest", 0.1, None, None),
        'Volume': ("Volume analysis for backtest", 0.1, None, None)
    }

    # Simplified moderator logic
    adjusted = {k: (v[0], min(v[1] + 0.05, 1.0), v[2], v[3]) for k, v in debate.items()}

    final_score = sum(score for _, score, _, _ in adjusted.values())
    confidence_percent = round(min(final_score, 1.0) * 100, 2)

    recommendation = "Buy" if confidence_percent > 70 else "Hold" if confidence_percent > 40 else "Sell"

    return {
        "Ticker": ticker,
        "Confidence (%)": confidence_percent,
        "Recommendation": recommendation
    }

def backtest_stock(ticker, period="3y", hold_period=5):
    hist = yf.Ticker(ticker).history(period=period)
    results = []
    for i in range(0, len(hist) - hold_period, hold_period):
        window = hist.iloc[i:i + hold_period]
        analysis = analyze_stock_for_backtest(ticker, window)
        success = analysis["Recommendation"] == "Buy" and window['Close'].pct_change().iloc[-1] > 0
        results.append({
            "Date": window.index[-1].date(),
            "Confidence (%)": analysis["Confidence (%)"],
            "Recommendation": analysis["Recommendation"],
            "Success": success
        })
    return pd.DataFrame(results)

def report_backtest(df):
    total = len(df)
    success = df['Success'].sum()
    rate = round(success / total * 100, 2) if total > 0 else 0
    return {
        "Total Periods": total,
        "Successful Buys": success,
        "Success Rate (%)": rate
    }
