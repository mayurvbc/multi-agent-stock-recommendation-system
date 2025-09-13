import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# --- Load FinBERT ---
@st.cache_resource(ttl=3600)
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

finbert_tokenizer, finbert_model = load_finbert()

# --- Helper Functions ---
def format_entry_exit(d):
    if not isinstance(d, dict):
        return {}
    formatted = {}
    for k, v in d.items():
        if v and isinstance(v, tuple) and v[0] is not None and v[1] is not None:
            formatted[k] = (round(float(v[0]), 2), round(float(v[1]), 2))
    return formatted

def format_debate(d):
    if not isinstance(d, dict):
        return "No debate data"
    formatted = [f"{k}: {v}" for k, v in d.items() if v is not None]
    return "\n".join(formatted)

# --- Stock tickers ---
tickers_pool = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS',
    'LT.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'AXISBANK.NS', 'BHARTIARTL.NS', 'MARUTI.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'TECHM.NS', 'WIPRO.NS', 'SUNPHARMA.NS', 'ADANIENT.NS',
    'JSWSTEEL.NS', 'ONGC.NS', 'TATAMOTORS.NS', 'ULTRACEMCO.NS', 'HCLTECH.NS', 'BPCL.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'VEDL.NS', 'TITAN.NS', 'INDUSINDBK.NS', 'NTPC.NS'
]

# --- Fetch stock data ---
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="6mo"):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period=period)
        info = data.info
        if len(hist) < 30:
            return None, None
        return hist, info
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def get_top_stocks_yf(tickers, n=5):
    stock_volumes = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="1mo")
            avg_volume = data['Volume'].mean()
            stock_volumes.append((ticker, avg_volume))
        except:
            continue
    stock_volumes.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in stock_volumes[:n]]

@st.cache_data(ttl=3600)
def get_top_stocks(n=5):
    return get_top_stocks_yf(tickers_pool, n)

# --- Agents ---
def price_action_agent(hist):
    ma10 = hist['Close'].rolling(10).mean().iloc[-1]
    ma30 = hist['Close'].rolling(30).mean().iloc[-1]
    last = hist['Close'].iloc[-1]
    atr = hist['Close'].diff().abs().rolling(14).mean().iloc[-1]
    if ma10 > ma30:
        target = last + atr * 2
        timeline = "Short-term"
        return "Bullish MA", 0.5, last * 1.01, last * 1.05, target, timeline
    else:
        target = last - atr * 2
        timeline = "Short-term"
        return "Bearish MA", 0.1, last * 0.99, last * 0.95, target, timeline

def technical_agent_enhanced(hist):
    close = hist['Close']
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    macd_signal = macd.iloc[-1] - signal.iloc[-1]
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    target_price = upper.iloc[-1] if macd_signal > 0 else lower.iloc[-1]
    trend_strength = abs(macd_signal)
    timeline = "Short-term" if trend_strength < 0.5 else "Medium-term"
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    rsi = 100 - 100 / (1 + rs)
    rsi_signal = 1 if rsi.iloc[-1] < 30 else -0.5 if rsi.iloc[-1] > 70 else 0
    score = 0.2 + 0.1 * (macd_signal > 0) + 0.05 * (close.iloc[-1] < upper.iloc[-1]) + 0.05 * rsi_signal
    score = min(max(score, 0), 1.0)
    return f"MACD={round(macd.iloc[-1], 2)}, RSI={round(rsi.iloc[-1], 1)}", score, None, None, target_price, timeline

def fundamental_agent_enhanced(info):
    pe = info.get('trailingPE', None)
    roe = info.get('returnOnEquity', None)
    de = info.get('debtToEquity', None)
    epsg = info.get('earningsQuarterlyGrowth', None)
    div = info.get('dividendYield', None)
    score = 0.1
    args = []
    if pe and pe < 30:
        score += 0.05
        args.append(f"PE={pe}")
    elif pe:
        score -= 0.05
    if roe and roe > 0.1:
        score += 0.05
        args.append(f"ROE={round(roe, 2)}")
    elif roe:
        score -= 0.05
    if de and de < 1:
        score += 0.05
        args.append(f"D/E={de}")
    elif de:
        score -= 0.05
    if epsg and epsg > 0.05:
        score += 0.05
        args.append(f"EPSG={round(epsg, 2)}")
    elif epsg:
        score -= 0.05
    if div:
        score += 0.05
        args.append(f"DivYield={round(div, 2)}")
    score = min(max(score, 0), 1.0)
    return " | ".join(args) if args else "Moderate fundamentals", score, None, None, None, "Medium-term"

def volume_agent(hist):
    v10 = hist['Volume'].rolling(10).mean().iloc[-1]
    v30 = hist['Volume'].rolling(30).mean().iloc[-1]
    if v10 > 1.5 * v30:
        return "Volume spike", 0.1, None, None, None, "Short-term"
    else:
        return "Volume normal", 0.0, None, None, None, "Short-term"

def sentiment_agent_financial(headlines):
    scores = []
    for text in headlines:
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = finbert_model(**inputs)
        probs = softmax(outputs.logits.detach().numpy()[0])
        scores.append(probs[1] - probs[2])  # Positive - Negative
    if scores:
        avg_score = sum(scores) / len(scores)
        score = min(max(0.25 + avg_score * 0.25, 0), 1.0)
        return f"{len([s for s in scores if s > 0])}/{len(scores)} positive", score, None, None, None, "Short-term"
    else:
        return "No news", 0.25, None, None, None, "Short-term"

def moderator(debate_dict):
    scores = [score for _, score, _, _, _, _ in debate_dict.values()]
    disagreement = max(scores) - min(scores)
    adjusted = {}
    for agent, (arg, score, entry, exit_level, target, timeline) in debate_dict.items():
        if disagreement > 0.3 and score == max(scores):
            adjusted[agent] = (arg + " (boost)", min(score + 0.05, 1.0), entry, exit_level, target, timeline)
        else:
            adjusted[agent] = (arg, score, entry, exit_level, target, timeline)
    return adjusted

# --- Analyze stock ---
def analyze_stock(ticker):
    try:
        hist, info = fetch_stock_data(ticker)
        if hist is None or info is None:
            return {
                "Ticker": ticker,
                "Recommendation": "Data Unavailable",
                "Confidence (%)": 0,
                "Risk Level": "N/A",
                "Debate Transcript": {},
                "Entry/Exit Levels": {},
                "Current Price": None,
                "Target Price": None,
                "Suggested Timeline": None,
                "Latest Data": None
            }
        
        last_price = hist['Close'].iloc[-1]
        latest_time = hist.index[-1]
        # Ensure latest_time is timezone-aware
        if latest_time.tzinfo is None:
            latest_time = latest_time.tz_localize('UTC')
        else:
            latest_time = latest_time.tz_convert('UTC')
        
        debate = {}
        debate['Price Action'] = price_action_agent(hist)
        debate['Technical'] = technical_agent_enhanced(hist)
        
        try:
            news_items = yf.Ticker(ticker).news[:5]
            headlines = [n['title'] for n in news_items if 'title' in n]
        except:
            headlines = []
        debate['Sentiment'] = sentiment_agent_financial(headlines)
        
        debate['Fundamentals'] = fundamental_agent_enhanced(info)
        debate['Volume'] = volume_agent(hist)
        
        adjusted = moderator(debate)
        agent_weights = [0.27, 0.23, 0.23, 0.18, 0.09]  # Sum to 1.0
        final_score = sum(w * score for w, (_, score, _, _, _, _) in zip(agent_weights, adjusted.values()))
        
        volatility = hist['Close'].pct_change().rolling(30).std().iloc[-1]
        # Adjust for data freshness
        now_utc = datetime.now(timezone.utc)
        data_age = (now_utc - latest_time).total_seconds() / 3600
        freshness_factor = 1.0  # Temporary for debugging
        final_score_normalized = min(final_score * max(0, 1 - 0.5 * volatility) * freshness_factor, 1.0)
        confidence_percent = round(final_score_normalized * 100, 2)
        
        # Log scores for debugging
        for agent, (arg, score, _, _, _, _) in adjusted.items():
            print(f"{ticker} - {agent}: Score = {score}, Arg = {arg}")
        print(f"{ticker} - Final Score: {final_score}, Volatility: {volatility}, Freshness: {freshness_factor}, Normalized: {final_score_normalized}")
        
        if final_score_normalized >= 0.6:
            recommendation = "Buy"
            risk_level = "High-risk"
        elif final_score_normalized >= 0.3:
            recommendation = "Hold"
            risk_level = "Medium-risk"
        else:
            recommendation = "Sell"
            risk_level = "Low-risk"
        
        top_agent = max(adjusted.items(), key=lambda x: x[1][1])
        target_price = top_agent[1][4]
        timeline = top_agent[1][5]
        
        transcript = {agent: arg for agent, (arg, _, _, _, _, _) in adjusted.items()}
        entry_exit = {agent: (entry, exit_level) for agent, (_, _, entry, exit_level, _, _) in adjusted.items() if entry and exit_level}
        
        return {
            "Ticker": ticker,
            "Current Price": round(last_price, 2),
            "Recommendation": recommendation,
            "Confidence (%)": confidence_percent,
            "Risk Level": risk_level,
            "Target Price": round(target_price, 2) if target_price else None,
            "Suggested Timeline": timeline,
            "Debate Transcript": transcript,
            "Entry/Exit Levels": entry_exit,
            "Latest Data": latest_time.strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        st.warning(f"Error analyzing {ticker}: {str(e)}")
        return {
            "Ticker": ticker,
            "Recommendation": "Data Unavailable",
            "Confidence (%)": 0,
            "Risk Level": "N/A",
            "Debate Transcript": {},
            "Entry/Exit Levels": {},
            "Current Price": None,
            "Target Price": None,
            "Suggested Timeline": None,
            "Latest Data": None
        }

# --- Streamlit UI ---
st.title("Multi-Agent Stock Recommendation System (Phase 8 Upgrade)")

# Allow custom ticker input
custom_ticker = st.text_input("Enter a custom ticker (e.g., RELIANCE.NS) or leave blank to analyze top stocks:")
if custom_ticker:
    tickers_to_analyze = [custom_ticker]
else:
    tickers_to_analyze = get_top_stocks(n=5)

if st.button("Generate Recommendations"):
    top_results = [analyze_stock(t) for t in tickers_to_analyze]
    df_top = pd.DataFrame(top_results)
    
    if not df_top.empty:
        df_top['Debate Transcript'] = df_top['Debate Transcript'].apply(format_debate)
        df_top['Entry/Exit Levels'] = df_top['Entry/Exit Levels'].apply(format_entry_exit)
        
        st.subheader("Stock Analysis")
        st.dataframe(
            df_top[['Ticker', 'Current Price', 'Target Price', 'Suggested Timeline', 'Recommendation', 'Confidence (%)', 'Risk Level', 'Latest Data']],
            column_config={
                "Confidence (%)": st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        st.markdown("---")
        st.subheader("Detailed Analysis")
        for i, row in df_top.iterrows():
            with st.expander(f"{row['Ticker']} - {row['Recommendation']} ({row['Confidence (%)']}%, {row['Risk Level']})"):
                st.text(f"Current Price: {row['Current Price']}")
                st.text(f"Target Price: {row['Target Price']} ({row['Suggested Timeline']})")
                st.text("Debate Transcript:\n" + row['Debate Transcript'])
                st.text("Entry/Exit Levels:\n" + str(row['Entry/Exit Levels']))
                st.text(f"Latest Data Fetched: {row['Latest Data']}")
                
                # Plot price history
                hist, _ = fetch_stock_data(row['Ticker'])
                if hist is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    hist['Close'].plot(ax=ax, label='Close Price')
                    hist['Close'].rolling(10).mean().plot(ax=ax, label='MA10')
                    hist['Close'].rolling(30).mean().plot(ax=ax, label='MA30')
                    ax.legend()
                    ax.set_title(f"{row['Ticker']} Price History")
                    st.pyplot(fig)
        
        # Analyze remaining tickers if no custom ticker
        if not custom_ticker:
            remaining_tickers = [t for t in tickers_pool if t not in tickers_to_analyze]
            remaining_results = [analyze_stock(t) for t in remaining_tickers]
            df_remaining = pd.DataFrame(remaining_results)
            
            if not df_remaining.empty:
                df_remaining['Debate Transcript'] = df_remaining['Debate Transcript'].apply(format_debate)
                df_remaining['Entry/Exit Levels'] = df_remaining['Entry/Exit Levels'].apply(format_entry_exit)
                
                st.markdown("---")
                st.subheader("Confidence Ratings - Other Stocks")
                st.dataframe(
                    df_remaining[['Ticker', 'Current Price', 'Target Price', 'Suggested Timeline', 'Recommendation', 'Confidence (%)', 'Risk Level', 'Latest Data']],
                    column_config={
                        "Confidence (%)": st.column_config.NumberColumn(format="%.2f")
                    }
                )

    # Download button
    if not df_top.empty:
        csv = df_top.to_csv(index=False)
        st.download_button(
            label="Download Top Stocks Data as CSV",
            data=csv,
            file_name="stock_recommendations.csv",
            mime="text/csv"
        )
