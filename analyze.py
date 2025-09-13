import yfinance as yf
from agents import price_action_agent, technical_agent_enhanced, fundamental_agent_enhanced, volume_agent, sentiment_agent_financial, moderator
import pandas as pd

def analyze_stock(ticker):
    hist = yf.Ticker(ticker).history(period="6mo")
    info = yf.Ticker(ticker).info
    headlines = [f"{ticker} earnings highlights", f"{ticker} regulatory risks"]

    debate = {
        'Price Action': price_action_agent(hist),
        'Technical': technical_agent_enhanced(hist),
        'Sentiment': sentiment_agent_financial(headlines),
        'Fundamentals': fundamental_agent_enhanced(info),
        'Volume': volume_agent(hist)
    }

    adjusted = moderator(debate)

    final_score = sum(score for _, score, _, _ in adjusted.values())
    confidence_percent = round(min(final_score, 1.0) * 100, 2)

    recommendation = "Buy" if confidence_percent > 70 else "Hold" if confidence_percent > 40 else "Sell"
    risk_level = "High-risk" if confidence_percent > 70 else "Medium-risk" if confidence_percent > 40 else "Low-risk"

    current_price = hist['Close'].iloc[-1]
    target_price = round(current_price * 1.1 if recommendation == "Buy" else current_price * 0.9, 2)

    return {
        "Ticker": ticker,
        "Current Price": round(current_price, 2),
        "Target Price": target_price,
        "Suggested Timeline": "1-3 months",
        "Recommendation": recommendation,
        "Confidence (%)": confidence_percent,
        "Risk Level": risk_level,
        "Debate Transcript": {k: v[0] for k, v in adjusted.items()},
        "Entry/Exit Levels": {k: (None, None) for k in adjusted},
        "Latest Data": str(hist.index[-1].date())
    }

def analyze_stock_for_backtest(ticker, hist_window):
    info = yf.Ticker(ticker).info
    return {"Ticker": ticker, "Confidence (%)": 60, "Recommendation": "Buy"}

def backtest_stock(ticker, period="3y", hold_period=5):
    hist = yf.Ticker(ticker).history(period=period)
    results = []
    for i in range(0, len(hist)-hold_period, hold_period):
        window = hist.iloc[i:i+hold_period]
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
    return {"Total Periods": total, "Successful Buys": success, "Success Rate (%)": rate}