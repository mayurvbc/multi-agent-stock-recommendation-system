def price_action_agent(hist):
    return "Price action analysis", 0.2, None, None

def technical_agent_enhanced(hist):
    return "Technical analysis", 0.2, None, None

def fundamental_agent_enhanced(info):
    return "Fundamental analysis", 0.2, None, None

def volume_agent(hist):
    return "Volume analysis", 0.1, None, None

def sentiment_agent_financial(headlines):
    return "Sentiment analysis", 0.1, None, None

def moderator(debate_dict):
    return {k: (v[0], min(v[1]+0.05,1.0), v[2], v[3]) for k,v in debate_dict.items()}