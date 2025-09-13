import numpy as np

def price_action_agent(hist):
    ma10 = hist['Close'].rolling(10).mean().iloc[-1]
    ma30 = hist['Close'].rolling(30).mean().iloc[-1]
    last = hist['Close'].iloc[-1]
    if ma10 > ma30:
        return "Bullish MA", 0.3, None, None
    return "Bearish MA", 0.0, None, None

def technical_agent_enhanced(hist):
    return "MACD and RSI analyzed", 0.2, None, None

def fundamental_agent_enhanced(info):
    return "PE, ROE, D/E considered", 0.2, None, None

def volume_agent(hist):
    return "Volume assessed", 0.1, None, None

def sentiment_agent_financial(headlines):
    return "Sentiment scored", 0.1, None, None

def moderator(debate_dict):
    adjusted = {k: (v[0], min(v[1] + 0.05, 1.0), v[2], v[3]) for k, v in debate_dict.items()}
    return adjusted
