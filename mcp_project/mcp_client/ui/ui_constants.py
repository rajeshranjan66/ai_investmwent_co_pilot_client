# ui_constants.py
import streamlit
"""Constants for the Stock Analysis UI"""
# UI Styles for different components
# UI Styles
UI_STYLES = {
    "main_container": """
        <div style='
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            margin: 1rem 0;
        '>
    """,
    "header": """
        <div style='
            text-align: center;
            color: darkblue;
            padding: 1rem;
            margin-bottom: 1rem;
        '>
    """,
    "response_container": """
        <div style='
            padding: 1rem;
            background-color: white;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin: 1rem 0;
        '>
    """,
    "sidebar_container": """
        <div style='
            padding: 0.5rem;
            background-color: #f1f3f4;
            border-radius: 0.3rem;
            margin: 0.5rem 0;
        '>
    """
}

# In mcp_client/ui/ui_constants.py

# A selection of tickers from different markets supported by Yahoo Finance
MARKET_STOCKS = {
    "🇺🇸 USA": {
        "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet (Google)",
        "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "NVIDIA", "JPM": "JPMorgan Chase",
        "V": "Visa", "C": "Citigroup"
    },
    "🇸🇬 Singapore": {
        "D05.SI": "DBS Group", "O39.SI": "OCBC Bank", "U11.SI": "UOB",
        "C6L.SI": "Singapore Airlines", "Z74.SI": "Singtel"
    },
    "🇮🇳 India": {
        "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy",
        "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "HINDUNILVR.NS": "Hindustan Unilever"
    }
}

TOP_STOCKS = {
    "RGTI": "Rigetti Computing Inc",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "C": "Citigroup",
    "JPM": "JP Morgan",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
    "V": "Visa",
    "MA": "Mastercard",
    "D05.SI": "DBS",
    "O39.SI": "OCBC",
    "U11.SI": "UOB",
    "HSBC": "HSBC Holdings",
    "BABA": "Alibaba",
    "TCEHY": "Tencent",
    "NFLX": "Netflix",
    "DIS": "Disney",
    "INTC": "Intel",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "PYPL": "PayPal",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "PEP": "PepsiCo",
    "KO": "Coca-Cola",
    "PG": "Procter & Gamble",
    "UNH": "UnitedHealth",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "PFE": "Pfizer",
    "MRK": "Merck",
    "T": "AT&T",
    "VZ": "Verizon",
    "SBUX": "Starbucks",
    "MCD": "McDonald's"
}

# = [1, 5, 10, 20, 50, 100, 150, 200, 500, 1000, 5000]
#ANALYSIS_YEARS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]

ANALYSIS_PROMPTS = {
    #"dividend": "Using historical dividend data, company financials, and recent market trends, calculate the projected total dividends for {quantity} shares of {stock} over the next {years} years. Include: (1) a summary of the company’s recent dividend history, (2) an estimate of future dividend growth based on past patterns, (3) the projected annual and cumulative dividends, and (4) any relevant financial metrics or news that could impact future dividends. Present your analysis with clear calculations and a brief explanation of your assumptions.",
    # "dividend": (
    #     "Analyze dividend projection for the following stocks:\n"
    #     "{portfolio_details}\n\n"
    #     "Instructions:\n"
    #     "- Use historical dividend data, company financials, and recent market trends.\n"
    #     "- Calculate projected total dividends for each holding over its specified period.\n"
    #     "- Include:\n"
    #     "  1. Summary of each company’s recent dividend history.\n"
    #     "  2. Estimate of future dividend growth based on past patterns.\n"
    #     "  3. Projected annual and cumulative dividends.\n"
    #     "  4. Relevant financial metrics or news that could impact future dividends.\n"
    #     "- Present your analysis with clear calculations and a brief explanation of your assumptions."
    # ),
    "dividend": (
        "Role: You are a financial analyst specializing in dividend forecasting for stock portfolios.\n\n"
        "Task: Project total dividends for the following holdings:\n"
        "{portfolio_details}\n\n"
        "Instructions:\n"
        "- Use historical dividend data, company financials, and recent market trends.\n"
        "- Calculate projected total dividends for each holding over its specified period.\n"
        "- Include:\n"
        "  1. Summary of each company’s recent dividend history.\n"
        "  2. Estimate of future dividend growth based on past patterns.\n"
        "  3. Projected annual and cumulative dividends.\n"
        "  4. Relevant financial metrics or news that could impact future dividends.\n"
        "- Present your analysis with clear calculations and a brief explanation of your assumptions.\n\n"
        "Output:\n"
        "- Table of projected dividends for each stock.\n"
        "- Brief commentary on key factors affecting dividend projections."
    ),
    #"return": "Using historical price data, dividend history, and key financial metrics for {stock}, calculate the potential total return for an investment of {quantity} shares held over {years} years.Include: (1) the initial investment amount, (2) annual and cumulative price appreciation,(3) total dividends received (assume dividends are reinvested if relevant), (4) the final investment value, and (5) the annualized rate of return (CAGR).Reference any significant financial events or news in this period that may have impacted returns. Show all calculations step by step and clearly state any assumptions.",
    # "return": (
    #     "Analyze total return projection for the following stocks:\n"
    #     "{portfolio_details}\n\n"
    #     "Instructions:\n"
    #     "- Use historical price data, dividend history, and key financial metrics.\n"
    #     "- Calculate the potential total return for each holding over its specified period.\n"
    #     "- Include:\n"
    #     "  1. Initial investment amount for each stock.\n"
    #     "  2. Annual and cumulative price appreciation.\n"
    #     "  3. Total dividends received (assume reinvestment if relevant).\n"
    #     "  4. Final investment value.\n"
    #     "  5. Annualized rate of return (CAGR).\n"
    #     "- Reference any significant financial events or news in this period that may have impacted returns.\n"
    #     "- Show all calculations step by step and clearly state any assumptions."
    # ),
    "return": (
        "Role: You are a financial analyst specializing in total return analysis for stock portfolios.\n\n"
        "Task: Calculate the potential total return for the following holdings:\n"
        "{portfolio_details}\n\n"
        "Instructions:\n"
        "- Use historical price data, dividend history, and key financial metrics.\n"
        "- Calculate the potential total return for each holding over its specified period.\n"
        "- Include:\n"
        "  1. Initial investment amount for each stock.\n"
        "  2. Annual and cumulative price appreciation.\n"
        "  3. Total dividends received (assume reinvestment if relevant).\n"
        "  4. Final investment value.\n"
        "  5. Annualized rate of return (CAGR).\n"
        "- Reference any significant financial events or news in this period that may have impacted returns.\n"
        "- Show all calculations step by step and clearly state any assumptions.\n\n"
        "Output:\n"
        "- Table of total returns for each stock.\n"
        "- Brief commentary on performance and influencing factors."
    ),
    #"detailed": "Using all available financial data tools and web search, provide a comprehensive analysis of {stock}. Include: (1) a summary of the company's business and recent news or events (using web search if needed),technical analysis based on historical price trends and key statistics (high/low/average prices, volatility, major movements),detailed financial health review with annual and quarterly financials (revenue, net income, EPS, margins, debt, cash flow),calculation and interpretation of important ratios (P/E, ROE, ROA, debt/equity, etc.),analysis of dividend history and projected future dividends, summary of major institutional holders and their recent activity,review of recent earnings reports and calendar events, overall market sentiment and analyst opinions from the latest news, and (9) a concise summary of the company’s strengths, risks, and investment outlook. Present findings clearly, using tables or charts where appropriate, and show all calculations and reasoning.",
    "detailed": (
        "Role: You are a financial analyst specializing in comprehensive stock analysis for portfolios.\n\n"
        "Task: Provide a detailed analysis for the following holdings:\n"
        "{portfolio_details}\n\n"
        "Instructions:\n"
        "- Use all available financial data tools and web search.\n"
        "- For each holding, include:\n"
        "  1. Summary of the company's business and recent news/events.\n"
        "  2. Technical analysis of historical price trends and key statistics (high/low/average prices, volatility, major movements).\n"
        "  3. Financial health review with annual and quarterly financials (revenue, net income, EPS, margins, debt, cash flow).\n"
        "  4. Calculation and interpretation of important ratios (P/E, ROE, ROA, debt/equity, etc.).\n"
        "  5. Analysis of dividend history and projected future dividends.\n"
        "  6. Summary of major institutional holders and their recent activity.\n"
        "  7. Review of recent earnings reports and calendar events.\n"
        "  8. Overall market sentiment and analyst opinions from the latest news.\n"
        "  9. Concise summary of strengths, risks, and investment outlook.\n"
        "- Present findings clearly, using tables or charts where appropriate, and show all calculations and reasoning.\n\n"
        "Output:\n"
        "- Table summarizing key metrics for each stock.\n"
        "- Brief commentary on each company’s outlook and risks."
    ),
    #"price_history" : "Retrieve and analyze the historical price data for {stock} over the past {years} years using available financial data tools. include: (1) a summary of price trends and key statistics (such as highest, lowest, and average prices), (2) identification of major price movements or volatility periods, (3) reference any significant company or market events (using recent news if available) that may have influenced the price, and (4) provide a visual summary (such as a table or chart) if possible. Explain your findings clearly and highlight any patterns or anomalies in the price history.",
    "price_history": (
        "Role: You are a financial analyst specializing in historical price analysis for stock portfolios.\n\n"
        "Task: Retrieve and analyze historical price data for the following holdings:\n"
        "{portfolio_details}\n\n"
        "Instructions:\n"
        "- Use available financial data tools.\n"
        "- For each holding, include:\n"
        "  1. Summary of price trends and key statistics (highest, lowest, average prices).\n"
        "  2. Identification of major price movements or volatility periods.\n"
        "  3. Reference any significant company or market events (using recent news if available) that may have influenced the price.\n"
        "  4. Provide a visual summary (table or chart) if possible.\n"
        "- Explain findings clearly and highlight any patterns or anomalies in the price history.\n\n"
        "Output:\n"
        "- Table or chart summarizing price history for each stock.\n"
        "- Brief commentary on notable trends, events, and anomalies."
    ),
    #"financials": "Using all available financial data tools, analyze the key financial metrics and ratios for {stock} over the past {years} years. Include: (1) a summary table of annual and quarterly financials (revenue, net income, EPS, margins, debt, cash flow, etc.),(2) calculation and interpretation of important ratios (P/E, ROE, ROA, debt/equity, etc.),(3) trends and significant changes in financial health,(4) a summary of major institutional holders and their recent activity,(5) reference any notable earnings reports or events, and (6) highlight relevant recent news or market sentiment that may impact the company’s financial outlook.  Present your findings clearly, using tables or charts where appropriate, and provide a concise summary of the company’s financial strengths and risks.",
    "financials": (
        "Role: You are a financial analyst specializing in financial metrics and ratio analysis for stock portfolios.\n\n"
        "Task: Analyze key financial metrics and ratios for the following holdings:\n"
        "{portfolio_details}\n\n"
        "Instructions:\n"
        "- Use all available financial data tools.\n"
        "- For each holding, include:\n"
        "  1. Summary table of annual and quarterly financials (revenue, net income, EPS, margins, debt, cash flow, etc.).\n"
        "  2. Calculation and interpretation of important ratios (P/E, ROE, ROA, debt/equity, etc.).\n"
        "  3. Trends and significant changes in financial health.\n"
        "  4. Summary of major institutional holders and their recent activity.\n"
        "  5. Reference any notable earnings reports or events.\n"
        "  6. Highlight relevant recent news or market sentiment that may impact the company’s financial outlook.\n"
        "- Present your findings clearly, using tables or charts where appropriate.\n\n"
        "Output:\n"
        "- Table summarizing key financial metrics and ratios for each stock.\n"
        "- Brief commentary on each company’s financial strengths and risks."
    ),
    #"news": "Using web search and financial data tools, retrieve and analyze the latest 5 news articles and market sentiment for {stock}.For each article, summarize the key points, assess the sentiment (positive, negative, or neutral), and explain how it may impact {stock}'s price or outlook.Include any major recent events, product launches, regulatory changes, or analyst opinions. Conclude with an overall sentiment summary and highlight any actionable insights for investors.",
    # "news": (
    #     "Role: You are a financial analyst specializing in news and sentiment analysis for stock portfolios.\n\n"
    #     "Task: Retrieve and analyze the latest news and market sentiment for the following holdings:\n"
    #     "{portfolio_details}\n\n"
    #     "Instructions:\n"
    #     "- Use web search and financial data tools.\n"
    #     "- For each holding, include:\n"
    #     "  1. The latest 5 news articles and market sentiment.\n"
    #     "  2. For each article, summarize key points and assess sentiment (positive, negative, or neutral).\n"
    #     "  3. Explain how each article may impact the stock's price or outlook.\n"
    #     "  4. Include major recent events, product launches, regulatory changes, or analyst opinions.\n"
    #     "- Conclude with an overall sentiment summary and highlight actionable insights for investors.\n\n"
    #     "Output:\n"
    #     "- Table summarizing news, sentiment, and impact for each stock.\n"
    #     "- Brief commentary on overall sentiment and actionable insights."
    # ),

    "news": (
        "Role: You are a financial analyst specializing in news and sentiment analysis for stock portfolios.\n\n"
        "Task: Retrieve and analyze the latest news and market sentiment for the following holdings:\n"
        "{portfolio_details}\n\n"
        "CRITICAL FORMATTING REQUIREMENTS:\n"
        "- You MUST output a properly formatted markdown table with exactly 5 columns\n"
        "- Table format: | # | Article & Source | Key Points | Sentiment | Potential Impact |\n"
        "- Each row must be on a single line - no line breaks within cells\n"
        "- Use pipe characters | to separate columns\n"
        "- Keep sentiment to single words: Positive, Negative, or Neutral\n"
        "- Key Points should be brief bullet points separated by semicolons, not new lines\n\n"
        "Instructions:\n"
        "- Use web search and financial data tools.\n"
        "- For each holding, include:\n"
        "  1. The latest 5 news articles and market sentiment.\n"
        "  2. For each article, summarize key points and assess sentiment (positive, negative, or neutral).\n"
        "  3. Explain how each article may impact the stock's price or outlook.\n"
        "  4. Include major recent events, product launches, regulatory changes, or analyst opinions.\n"
        "- Conclude with an overall sentiment summary and highlight actionable insights for investors.\n\n"
        "Output:\n"
        "- Properly formatted markdown table with columns: #, Article & Source, Key Points, Sentiment, Potential Impact\n"
        "- Brief commentary on overall sentiment and actionable insights."
    ),

    # "dividend_portfolio": (
    # "For the following portfolio, calculate the projected total dividends for each stock over its specified period:\n"
    # "{portfolio_details}\n"
    # "Each line is in the format: '<quantity> shares of <stock> for <years> years'.\n"
    # "Include: (1) a summary of each company’s recent dividend history, (2) an estimate of future dividend growth, "
    # "(3) projected annual and cumulative dividends for each stock and for the whole portfolio, and (4) any relevant financial metrics or news. "
    # "Present your analysis with clear calculations and a brief explanation of your assumptions."
    # ),
    "dividend_portfolio": (
        "Role: You are a financial analyst specializing in portfolio dividend forecasting.\n\n"
        "Task: Calculate the projected total dividends for the following portfolio holdings:\n"
        "{portfolio_details}\n\n"
        "Instructions:\n"
        "- Each line is in the format: '<quantity> shares of <stock> for <years> years'.\n"
        "- For each holding, include:\n"
        "  1. Summary of recent dividend history.\n"
        "  2. Estimate of future dividend growth based on past patterns.\n"
        "  3. Projected annual and cumulative dividends for each stock and the whole portfolio.\n"
        "  4. Relevant financial metrics or news that could impact future dividends.\n"
        "- Present your analysis with clear calculations and a brief explanation of your assumptions.\n\n"
        "Output:\n"
        "- Table of projected dividends for each stock and the portfolio.\n"
        "- Brief commentary on key factors affecting dividend projections."
    ),
    # "quant_analysis": (
    # "Using advanced quantitative tools, Perform quant-level portfolio analysis of the following stocks:\n"
    # "{stock_details}\n"
    # "Your analysis should include: (1) Detailed statistical metrics (mean returns, variance, standard deviation, and correlation matrix between stocks)",
    # " (2) Risk metrics such as beta relative to SPY, Sharpe ratio, and maximum drawdown.", "(3) Performance metrics including CAGR and annualized returns. ",
    # "(4) Portfolio diversification assessment and optimization suggestions if applicable., and (5) Clear and concise insights with tables and charts to illustrate the portfolio's performance and risks. ",
    # "Present the results clearly with tables, charts, and concise explanations of the findings.",
    # "Also, provide any recommendations or actionable next steps based on the quantitative analysis"
    # )
    "quant_analysis": (
    "Using advanced quantitative methods, perform a detailed, professional-grade portfolio analysis for the following holdings:\n"
    "{stock_details}\n\n"
    
    "Where possible, cover the following areas (skip any sections if data is unavailable):\n"
    "• Statistical overview – mean, variance, standard deviation, and correlations between holdings.\n"
    "• Risk assessment – beta vs. SPY benchmark, Sharpe ratio, maximum drawdown.\n"
    "• Performance review – CAGR, annualized returns, volatility.\n"
    "• Portfolio structure – diversification level, concentration risks, and possible optimization strategies.\n"
    "• Actionable insights – clear recommendations for improvement or rebalancing.\n\n"

    "Present the results in a well-structured narrative with supporting tables, charts, or bullet points where useful. "
    "Tables can include: summary statistics, equity curve over time, daily returns, sector exposure breakdown, and portfolio weights. "
    "Ensure any visual or tabular data is labeled clearly so it can be parsed if needed.\n\n"

    "Conclude with a concise, plain-English summary of the main findings."
   )



}



