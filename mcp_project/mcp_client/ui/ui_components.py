# ui_components.py
"""Streamlit UI components for the Stock Analysis application"""
import logging

import streamlit as st
from typing import Dict, Any, Tuple
from .ui_constants import TOP_STOCKS, ANALYSIS_PROMPTS
from mcp_client.ui.ui_constants import MARKET_STOCKS


def create_header():
    """Create the application header"""

    # st.markdown(
    #     """
    #     <h2 style='text-align: center; color: darkblue;'>
    #         Welcome to Your AI Investment Co-Pilot <sup style='font-size: 14px; color: red;'>Powered by LLM & MCP</sup>
    #     </h2>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.markdown("""
    <div style="text-align: left; padding: 1rem 0;">
        <h2 style="color: #1f77b4;">Welcome to Your AI Investment Co-Pilot</h2>
        <p style="color: #666; font-size: 1.1rem;">
            Multi-Agent Stock Research & Forecasting Suite
        </p>
    </div>
    """, unsafe_allow_html=True)



def configure_page_settings():
    """Configure streamlit page settings"""
    st.set_page_config(
        page_title="Stock Research DeepAgent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    # ADD THE hide_streamlit_style HERE


import streamlit as st
def inject_seo_metadata():

    st.markdown(
            """
            <!-- Metadata Block (Singapore-Optimized) -->
            <meta name="title" content="Stock Research Agent in Singapore | Rajesh Ranjan">
            <meta name="description" content="AI-powered Stock Research Agent offering stock market analysis, SGX insights, and investment strategies for Singapore investors.">
            <meta name="keywords" content="stock research agent Singapore, SGX stock analysis, investment advisor Singapore, stock market research Singapore, Singapore stock insights">
            <meta property="og:type" content="website">
            <meta property="og:url" content="https://www.rajeshranjan.click/">
            <meta property="og:title" content="Stock Research Agent in Singapore | Rajesh Ranjan">
            <meta property="og:description" content="Get AI-driven stock research and SGX investment strategies. Trusted stock research agent in Singapore.">
            <meta property="og:image" content="https://www.rajeshranjan.click/preview.png">
            <meta property="twitter:card" content="summary_large_image">
            <meta property="twitter:url" content="https://www.rajeshranjan.click/">
            <meta property="twitter:title" content="Stock Research Agent in Singapore | Rajesh Ranjan">
            <meta property="twitter:description" content="AI-powered Stock Research Agent offering real-time SGX stock market insights and investment strategies.">
            <meta property="twitter:image" content="https://www.rajeshranjan.click/preview.png">
            <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@type": "LocalBusiness",
              "name": "Stock Research Agent",
              "image": "https://www.rajeshranjan.click/preview.png",
              "@id": "https://www.rajeshranjan.click",
              "url": "https://www.rajeshranjan.click",
              "telephone": "+65-XXXX-XXXX",
              "address": {
                "@type": "PostalAddress",
                "streetAddress": "Your Street Address",
                "addressLocality": "Singapore",
                "postalCode": "XXXXX",
                "addressCountry": "SG"
              },
              "geo": {
                "@type": "GeoCoordinates",
                "latitude": "1.3521",
                "longitude": "103.8198"
              },
              "sameAs": [
                "https://www.linkedin.com/in/yourprofile",
                "https://twitter.com/yourprofile"
              ]
            }
            </script>
            """,
            unsafe_allow_html=True
    )


    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header[data-testid="stHeader"] {display: none;}
            .stDeployButton {display:none;}
            .main > div {padding-top: 0 !important; margin-top: 0 !important;}
            .block-container {padding-top: 0 !important; margin-top: 0 !important;}
            .element-container:first-child {margin-top: 0;}
            html, body, [data-testid="stAppViewContainer"] {margin: 0; padding: 0; height: 100%; overflow: auto;}
            </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def show_spinner_overlay():
    import streamlit as st
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        margin: 0 !important;
        padding: 0 !important;
        height: 100% !important;
        overflow: hidden !important;
    }
    .overlay-spinner {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        width: 100vw; height: 100vh;
        background: rgba(255,255,255,0.7);
        z-index: 99999 !important;
        display: flex; align-items: center; justify-content: center;
        pointer-events: all;
    }
    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #3498db;
        border-radius: 50%;
        width: 60px; height: 60px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg);}
        100% { transform: rotate(360deg);}
    }
    </style>
    <div class="overlay-spinner">
        <div class="loader"></div>
    </div>
    """, unsafe_allow_html=True)


# def set_custom_theme():
#     """
#     Applies a custom theme to the Streamlit app by injecting CSS styles.
#     """
#     st.markdown("""
#     <style>
#     /* General background and text styling */
#     .main {
#         background-color: #f5f5f5; /* Light gray background */
#         color: #333333; /* Dark gray text */
#     }
#
#     /* Header styling */
#     .stHeader {
#         background-color: #2e6bb7; /* Blue header background */
#         color: white; /* White text */
#         padding: 10px;
#         border-radius: 5px;
#     }
#
#     /* Button styling */
#     div[data-testid="stFormSubmitButton"] > button {
#         background-color: #2e6bb7 !important; /* Blue button */
#         color: white !important; /* White text */
#         border-radius: 5px !important;
#         padding: 10px 20px !important;
#         font-weight: bold !important;
#         border: none !important;
#         transition: background-color 0.3s ease !important;
#     }
#     div[data-testid="stFormSubmitButton"] > button:hover {
#         background-color: #235292 !important; /* Darker blue on hover */
#     }
#
#     /* Input field styling */
#     input, textarea {
#         border: 1px solid #cccccc !important; /* Light gray border */
#         border-radius: 5px !important;
#         padding: 10px !important;
#         font-size: 1rem !important;
#     }
#
#     /* Tab styling */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 20px;
#         justify-content: center;
#         margin-top: 10px;
#         margin-bottom: 10px;
#     }
#     .stTabs [data-baseweb="tab"] {
#         font-size: 16px;
#         font-weight: 500;
#         padding: 10px 15px;
#     }
#
#     /* Sidebar styling */
#     .sidebar .sidebar-content {
#         background-color: #ffffff; /* White background */
#         border-right: 1px solid #dddddd; /* Light gray border */
#     }
#     </style>
#     """, unsafe_allow_html=True)

def set_custom_theme():
    # Option 1: Professional Blue Theme
    custom_theme = """
        <style>
        :root {
            --primary-color: #2E86C1;
            --primary-hover: #21618C;
            --background-color: #f0f2f6;
            --secondary-background-color: #ffffff;
            --text-color: #2C3E50;
            --font: 'Helvetica Neue', sans-serif;
        }
        /*# Option 1: Professional Blue Theme
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: var(--primary-hover);
            border: none;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        */

        /* Option 2: Modern Green Theme (commented out)
        .stButton>button {
            background-color: #27AE60;
            color: white;
        }

        .stButton>button:hover {
            background-color: #219A52;
        }
        */


        Option 3: Neutral Gray Theme (commented out)
        .stButton>button {
            background-color: #34495E;
            color: white;
        }

        .stButton>button:hover {
            background-color: #2C3E50;
        }


        .stTextInput>div>div>input {
            border-radius: 4px;
        }

        h1, h2, h3 {
            font-family: var(--font);
            color: var(--text-color);
        }

        /* Additional styling for better visual hierarchy */
        .stSelectbox label,
        .stTextInput label {
            color: var(--text-color);
            font-weight: 500;
        }

        .stTextInput>div>div>input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 1px var(--primary-color);
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--secondary-background-color);
        }

        /* Improve readability */
        p, li {
            color: var(--text-color);
            line-height: 1.6;
        }
        </style>
    """
    st.markdown(custom_theme, unsafe_allow_html=True)


def create_multi_stock_selection_ui():
    from .ui_constants import TOP_STOCKS
    if "stock_rows" not in st.session_state:
        st.session_state["stock_rows"] = 1
    if "stocks_data" not in st.session_state:
        st.session_state["stocks_data"] = [{} for _ in range(st.session_state["stock_rows"])]

    # Add Stock button - Disable if limit of 10 is reached
    add_stock_disabled = st.session_state["stock_rows"] >= 10
    if st.button("+ Add Stock", disabled=add_stock_disabled):
        st.session_state["stock_rows"] += 1
        st.session_state["stocks_data"].append({})
        st.rerun()

    delete_index = None
    stocks = []
    for i in range(st.session_state["stock_rows"]):
        cols = st.columns([3, 3, 3, 1])
        stock = cols[0].selectbox(
            label="📈 Select Stock" if i == 0 else f"Stock for row {i+1}",
            options=list(TOP_STOCKS.keys()),
            format_func=lambda x: f"{TOP_STOCKS[x]} ({x})",
            key=f"stock_{i}",
            label_visibility="visible" if i == 0 else "collapsed"
        )
        quantity = cols[1].number_input(
            label="🔢 Number of Shares" if i == 0 else f"Shares for row {i + 1}",
            value=10,
            min_value=1,
            step=100,
            key=f"qty_{i}",
            label_visibility="visible" if i == 0 else "collapsed"
        )
        years = cols[2].number_input(
            label="📅 Analysis Period (Years)" if i == 0 else f"Years for row {i+1}",
            min_value=0, value=0, step=1, key=f"years_{i}",
            label_visibility="visible" if i == 0 else "collapsed"
        )
        # Add delete button for all rows except the first
        if i > 0:
            if cols[3].button("🗑️", key=f"delete_{i}"):
                delete_index = i
        stocks.append({"stock": stock, "quantity": quantity, "years": years})

    # Handle deletion after UI rendering
    if delete_index is not None:
        st.session_state["stock_rows"] -= 1
        # Remove the deleted row's widgets from session state
        for key in [f"stock_{delete_index}", f"qty_{delete_index}", f"years_{delete_index}"]:
            if key in st.session_state:
                del st.session_state[key]
        # Remove the row from stocks
        stocks.pop(delete_index)
        st.rerun()
    st.markdown(
        """
        <div style="display: flex; align-items: center; font-size: 14px; margin-bottom: 10px;">
            <span style="margin-right: 10px;">Add up to 10 stocks for portfolio analysis.</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
                """
                <div style="display: flex; align-items: center; font-size: 14px;">
                    <span>💵 Cash Available for Investment</span>
                    <span style="margin-left: 10px; font-size: 11px; color: gray;">
                        (represents the amount of cash or cash-equivalent value you currently hold outside of stocks)
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
    # Add cash input field aligned to the left
    cash_col = st.columns([1, 3, 3])[0]  # Use the first column for left alignment
    with cash_col:
        with cash_col:
            cash = st.number_input(
                "Enter Cash Amount",
                min_value=0.0,
                value=0.0,
                step=10000.0,
                key="cash_input",
                label_visibility="collapsed"  # Hide the label to save space
            )

    return stocks

# def create_multi_stock_selection_ui():
#     from .ui_constants import TOP_STOCKS
#     if "stock_rows" not in st.session_state:
#         st.session_state["stock_rows"] = 1
#     if "stocks_data" not in st.session_state:
#         st.session_state["stocks_data"] = [{} for _ in range(st.session_state["stock_rows"])]
#
#     # Add Stock button
#     if st.button("+ Add Stock") and st.session_state["stock_rows"] < 5:
#         st.session_state["stock_rows"] += 1
#         st.session_state["stocks_data"].append({})
#
#     delete_index = None
#     stocks = []
#     for i in range(st.session_state["stock_rows"]):
#         cols = st.columns([3, 3, 3, 1])
#         stock = cols[0].selectbox(
#             "📈 Select Stock" if i == 0 else "",
#             options=list(TOP_STOCKS.keys()),
#             format_func=lambda x: f"{TOP_STOCKS[x]} ({x})",
#             key=f"stock_{i}"
#         )
#         quantity = cols[1].number_input(
#             label="🔢 Number of Shares" if i == 0 else f"Shares for row {i + 1}",
#             value=10,
#             min_value=1,
#             step=100,
#             key=f"qty_{i}",
#             label_visibility="visible" if i == 0 else "collapsed"
#         )
#         # quantity = cols[1].number_input(
#         #     "🔢 Number of Shares" if i == 0 else "",
#         #     min_value=1, value=10, step=100, key=f"qty_{i}",
#         #)
#         years = cols[2].number_input(
#             "📅 Analysis Period (Years)" if i == 0 else "",
#
#             min_value=0, value=0, step=1, key=f"years_{i}",
#
#         )
#         # Add delete button for all rows except the first
#         if i > 0:
#             if cols[3].button("🗑️", key=f"delete_{i}"):
#                 delete_index = i
#         stocks.append({"stock": stock, "quantity": quantity, "years": years})
#
#     # Handle deletion after UI rendering
#     if delete_index is not None:
#         st.session_state["stock_rows"] -= 1
#         # Remove the deleted row's widgets from session state
#         for key in [f"stock_{delete_index}", f"qty_{delete_index}", f"years_{delete_index}"]:
#             if key in st.session_state:
#                 del st.session_state[key]
#         # Remove the row from stocks
#         stocks.pop(delete_index)
#         st.rerun()
#     st.markdown(
#         """
#         <div style="display: flex; align-items: center; font-size: 14px;">
#             <span>💵 Cash Holdings</span>
#             <span style="margin-left: 10px; font-size: 11px; color: gray;">
#                 (represents the amount of cash or cash-equivalent value you currently hold outside of stocks)
#             </span>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
#
#     # Add cash input field aligned to the left
#     cash_col = st.columns([2, 1, 1])[0]  # Use the first column for left alignment
#     with cash_col:
#         cash = st.number_input(
#             "Enter Cash Amount",
#             min_value=0.0,
#             value=0.0,
#             step=10000.0,
#             key="cash_input",
#             label_visibility="collapsed"  # Hide the label to save space
#         )
#
#     return stocks


def create_stock_selection_ui() -> Tuple[str, int, int]:
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_stock = st.selectbox(
            "📈 Select Stock",
            options=list(TOP_STOCKS.keys()),
            format_func=lambda x: f"{TOP_STOCKS[x]} ({x})"
        )

    with col2:
        selected_quantity = st.number_input(
            "🔢 Number of Shares",
            min_value=1,
            max_value=1000000,
            value=10,
            step=1
        )

    with col3:
        selected_years = st.number_input(
            "📅 Analysis Period (Years)",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )

    return selected_stock, int(selected_quantity), int(selected_years)

# In your function that creates the action buttons (e.g., create_action_buttons)

from typing import Dict


import streamlit as st
from typing import Dict


def create_analysis_buttons() -> Dict[str, bool]:
    """
    Creates and displays analysis action buttons with improved UI/UX.
    Returns a dictionary indicating which button was clicked.
    """
    # Inject custom CSS for the buttons
    st.markdown("""
    <style>
    
    /* Match Add Stock button to Express Analysis Actions bar */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(145deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }
    /* Analysis button styling */
    .analysis-btn {
        width: 100% !important;
        height: 80px !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        background: linear-gradient(145deg, #ffffff, #f8f9fa) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
        margin: 8px 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 10px !important;
    }

    .analysis-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        background: linear-gradient(145deg, #f8f9fa, #e9ecef) !important;
        border-color: #2e6bb7 !important;
    }

    .analysis-btn:active {
        transform: translateY(0) !important;
    }

    .btn-icon {
        font-size: 20px !important;
        margin-bottom: 5px !important;
    }

    .btn-text {
        font-size: 12px !important;
        font-weight: 600 !important;
        text-align: center !important;
        line-height: 1.2 !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #2e6bb7, #3b82f6);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 15px 0 10px 0;
        font-weight: 600;
        font-size: 16px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #2e6bb7 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    actions = {}

    st.markdown("---")
    st.markdown('<div class="section-header">🚀 Express Analysis Actions</div>', unsafe_allow_html=True)

    # Group 1: Performance & Projections
    with st.expander("📈 Performance & Projections", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            actions["return"] = st.button(
                "📊\nTotal Return",
                help="Project total return on investment",
                use_container_width=True,
                key="btn_return"
            )
        with col2:
            actions["dividend"] = st.button(
                "💰\nDividends",
                help="Project future dividend income",
                use_container_width=True,
                key="btn_dividend"
            )
        with col3:
            actions["price_history"] = st.button(
                "📈\nPrice History",
                help="Analyze historical price trends",
                use_container_width=True,
                key="btn_price_history"
            )

    # Group 2: Fundamental & News Analysis
    with st.expander("🔬 Fundamental & News Analysis", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            actions["financials"] = st.button(
                "📑\nFinancials",
                help="Review key financial statements and ratios",
                use_container_width=True,
                key="btn_financials"
            )
        with col2:
            actions["news"] = st.button(
                "📰\nNews/Sentiment",
                help="Analyze recent news and market sentiment",
                use_container_width=True,
                key="btn_news"
            )
        with col3:
            actions["detailed"] = st.button(
                "🔍\nDeep Dive",
                help="Get a comprehensive analysis of the stock",
                use_container_width=True,
                key="btn_detailed"
            )

    # Group 3: Portfolio-Level Analysis
    with st.expander("💼 Portfolio-Level Analysis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            actions["dividend_portfolio"] = st.button(
                "📦\nPortfolio Dividends",
                help="Forecast dividend income for the entire portfolio",
                use_container_width=True,
                key="btn_dividend_portfolio"
            )
        with col2:
            actions["quant_analysis"] = st.button(
                "📊\nQuant Analysis",
                help="Perform quantitative analysis on the portfolio",
                use_container_width=True,
                key="btn_quant_analysis"
            )

    return actions



# def create_analysis_buttons() -> Dict[str, bool]:
#     """
#     Creates and displays analysis action buttons grouped into expandable sections.
#     Returns a dictionary indicating which button was clicked.
#     """
#     actions = {}
#     st.markdown("---")
#     st.markdown("##### Express Analysis Actions")
#
#     # Group 1: Performance & Projections
#     with st.expander("📈 Performance & Projections", expanded=True):
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             actions["return"] = st.button("Total Return", help="Project total return on investment.", use_container_width=True)
#         with col2:
#             actions["dividend"] = st.button("Dividends", help="Project future dividend income.", use_container_width=True)
#         with col3:
#             actions["price_history"] = st.button("Price History", help="Analyze historical price trends.", use_container_width=True)
#
#     # Group 2: Fundamental & News Analysis
#     with st.expander("🔬 Fundamental & News Analysis"):
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             actions["financials"] = st.button("Financials", help="Review key financial statements and ratios.", use_container_width=True)
#         with col2:
#             actions["news"] = st.button("News/Sentiment", help="Analyze recent news and market sentiment.", use_container_width=True)
#         with col3:
#             actions["detailed"] = st.button("Deep Dive", help="Get a comprehensive analysis of the stock.", use_container_width=True)
#
#     # Group 3: Portfolio-Level Analysis
#     with st.expander("💼 Portfolio-Level Analysis"):
#         col1, col2 = st.columns(2)
#         with col1:
#             actions["dividend_portfolio"] = st.button("Portfolio Dividends", help="Forecast dividend income for the entire portfolio.", use_container_width=True)
#         with col2:
#             actions["quant_analysis"] = st.button("Quant Analysis", help="Perform quantitative analysis on the portfolio.", use_container_width=True)
#
#     return actions





def create_chat_interface() -> str:
    """Create the chat interface"""
    return st.text_area(
        "💬 Enter your stock research query:",
        value=(
            "What is the the current stock price of citi"),
        height=200
    )


# def display_agent_thinking(content: str):
#     """Display the agent's thinking process in the sidebar"""
#     st.sidebar.markdown(
#         '<div style="display: flex; align-items: center;">'
#         '<span style="font-size: 24px;">🤖</span>'
#         '<span style="font-weight: bold; margin-left: 10px;">How Re-Act Agent is leveraging MCP tools...</span>'
#         '</div>',
#         unsafe_allow_html=True
#     )
#     st.sidebar.markdown(f"```\n{content}\n```")


# def display_agent_thinking(thinking_content):
#     if thinking_content and thinking_content.strip():
#         with st.sidebar:
#             st.markdown("---")
#             with st.expander("🤔 Agent Thinking Process", expanded=False):
#                 # Use a container with fixed height to prevent layout shifts
#                 st.markdown("""
#                 <div style="max-height: 300px; overflow-y: auto; font-size: 0.8rem;
#                             background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
#                 {}
#                 </div>
#                 """.format(thinking_content.replace('\n', '<br>')), unsafe_allow_html=True)


def display_agent_thinking(thinking_content):
    if thinking_content and thinking_content.strip():
        with st.sidebar:
            # Add a container with fixed height for scrolling
            st.markdown("""
            <div style='max-height: 60vh; overflow-y: auto; border-bottom: 1px solid #e0e0e0; padding-bottom: 10px;'>
            """, unsafe_allow_html=True)

            with st.expander("🤔 Agent Thinking Process", expanded=False):
                st.markdown(f"""
                <div style="font-size: 0.8rem; background-color: #f8f9fa; 
                            padding: 10px; border-radius: 4px; max-height: 300px; overflow-y: auto;">
                {thinking_content.replace('\n', '<br>')}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

def format_analysis_prompts_for_rows(stock_rows, actions, analysis_prompts):
    prompts = []
    for action, clicked in actions.items():
        if clicked:
            if action == "quant_analysis":
                # Construct holdings dictionary from stock rows
                holdings = {r.get("stock", "Unknown"): r.get("quantity", 0) for r in stock_rows if "stock" in r}

                # Retrieve cash value entered by the user
                cash = st.session_state.get("cash_input", 0.0)
                holdings["CASH"] = cash

                # Format the prompt
                stock_details = ", ".join(
                    f"{stock} ({quantity} shares)" for stock, quantity in holdings.items() if stock != "CASH"
                )
                cash_details = f"Cash: ${cash}" if cash > 0 else "No cash holdings"

                prompt = analysis_prompts[action].format(
                    stock_details=f"{stock_details}, {cash_details}",

                )
                prompts.append((action, prompt, {"holdings": holdings}))

            elif action == "dividend_portfolio":
                portfolio_details = "\n".join(
                    f"{r.get('stock', 'Unknown')} ({r.get('quantity', 0)} shares)" for r in stock_rows
                )
                prompt = analysis_prompts[action].format(portfolio_details=portfolio_details)
                prompts.append((action, prompt, None))
                # In format_analysis_prompts_for_rows
            else:
                # Aggregate all selected stocks
                portfolio_details = "\n".join(
                    f"{r.get('stock', 'Unknown')} ({r.get('quantity', 0)} shares, {r.get('years', 0)} years)" for r in
                    stock_rows
                )
                prompt = analysis_prompts[action].format(portfolio_details=portfolio_details)
                prompts.append((action, prompt, stock_rows))

                # for row in stock_rows:
                #     prompt = analysis_prompts[action].format(
                #         stock=row.get("stock", "Unknown"),
                #         quantity=row.get("quantity", 0),
                #         years=row.get("years", 0)
                #     )
                #     prompts.append((action, prompt, row))

    return prompts




import re
import pandas as pd
import streamlit as st


import json


def render_forecast_table_from_text(content, key=None):
    """
    Display forecasted prices from either plain text or JSON (from MCP tool).
    """
    # Try to parse JSON first
    try:
        data = json.loads(content)
        if "forecast" in data:
            forecast = data["forecast"]
            dates = forecast.get("dates", [])
            prices = forecast.get("prices", [])
            cis = forecast.get("confidence_intervals", [])
            rows = []
            for i in range(len(dates)):
                ci = cis[i] if i < len(cis) else [None, None]
                rows.append({
                    "Date": dates[i],
                    "Price": prices[i] if i < len(prices) else None,
                    "95% CI Low": ci[0],
                    "95% CI High": ci[1]
                })
            if rows:
                df = pd.DataFrame(rows)
                st.write("Forecasted Prices with Confidence Intervals:")
                st.dataframe(df.head(5))
                with st.expander("Show all forecasted prices"):
                    st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download all rows as CSV", csv, "forecast.csv", "text/csv", key=key or "pdf_tab2")
            return
    except Exception:
        pass

    # Fallback: parse plain text (original logic)
    import re
    price_lines = re.findall(r'([A-Za-z]+\s\d{1,2},\s\d{4}):\s([\d.]+)', content)
    ci_lines = re.findall(r'([A-Za-z]+\s\d{1,2},\s\d{4}):\s\[(\d+\.\d+),\s*(\d+\.\d+)\]', content)
    prices = {date: float(price) for date, price in price_lines}
    cis = {date: (float(low), float(high)) for date, low, high in ci_lines}
    rows = []
    for date in prices:
        price = prices[date]
        ci = cis.get(date, (None, None))
        rows.append({
            "Date": date,
            "Price": price,
            "95% CI Low": ci[0],
            "95% CI High": ci[1]
        })
    if rows:
        df = pd.DataFrame(rows)
        st.write("Forecasted Prices with Confidence Intervals:")
        st.dataframe(df.head(5))
        with st.expander("Show all forecasted prices"):
            st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download all rows as CSV", csv, "forecast.csv", "text/csv", key=key or "pdf_tab2")


