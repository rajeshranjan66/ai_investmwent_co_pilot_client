
# https://github.com/langchain-ai/deepagents/blob/master/README.md
def fundamental_analyst(*args, **kwargs):
    print("************************Serving request:fundamental-analyst")
    return {
        "name": "fundamental-analyst",
        "description": "Performs deep fundamental analysis of companies including financial ratios, growth metrics, and valuation",
        "prompt": """You are an expert fundamental analyst with 15+ years of experience. 
        Focus on:
        - Financial statement analysis
        - Ratio analysis (P/E, P/B, ROE, ROA, Debt-to-Equity)
        - Growth metrics and trends
        - Industry comparisons
        - Intrinsic value calculations
        Always provide specific numbers and cite your sources."""
    }