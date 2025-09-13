def format_debate(d):
    if not isinstance(d, dict):
        return "No debate data"
    return "\n".join([f"{k}: {v}" for k,v in d.items()])

def format_entry_exit(d):
    if not isinstance(d, dict):
        return {}
    return {k: (str(v[0]), str(v[1])) for k,v in d.items() if v and isinstance(v, tuple)}