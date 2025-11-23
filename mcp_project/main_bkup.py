from datetime import time

import streamlit as st
from typing import Optional
import base64
import os
import sys
from dotenv import load_dotenv
from mcp_client.ui.ui_spinner import show_modal_spinner

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
load_dotenv()
import time


class SEOEntryPoints:
    def __init__(self):
        self.sample_stocks = [
            "Apple (AAPL)", "Microsoft (MSFT)", "Amazon (AMZN)", "Tesla (TSLA)",
            "NVIDIA (NVDA)", "Toyota (TM)", "Samsung (005930.KS)", "DBS Group (D05.SI)"
        ]

    def setup_page_config(self, page_type: str = "landing"):
        st.markdown("""
        <style>
        /* HIDE ONLY TOOLBAR ELEMENTS, KEEP SIDEBAR TOGGLE */
        /* Target and hide specific toolbar buttons */
        button[title="Deploy"],
        button[title="Get help"], 
        button[title="Report a bug"],
        button[title="Settings"],
        button[aria-label="Deploy"],
        button[aria-label="Get help"],
        button[aria-label="Report a bug"],
        button[aria-label="Settings"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }

        /* Hide toolbar containers that hold these buttons */
        div[data-testid="stToolbar"] > div:has(button[title="Deploy"]),
        div[data-testid="stToolbar"] > div:has(button[title="Get help"]),
        div[data-testid="stToolbar"] > div:has(button[title="Report a bug"]),
        div[data-testid="stToolbar"] > div:has(button[title="Settings"]) {
            display: none !important;
            visibility: hidden !important;
        }

        /* KEEP THE SIDEBAR TOGGLE BUTTON VISIBLE */
        button[title="Show sidebar"],
        button[title="Hide sidebar"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 15px !important;
            left: 15px !important;
            z-index: 1001 !important;
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            box-shadow: var(--card-shadow) !important;
            border: 1px solid var(--card-border) !important;
        }

        /* Ensure sidebar toggle stays on top */
        button[title="Show sidebar"],
        button[title="Hide sidebar"] {
            z-index: 9999 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        configs = {
            "landing": {
                "page_title": "AI Investment Co-Pilot | Multi-Agent Stock Analysis",
                "page_icon": "🚀",
                "meta_description": "Harness a multi-agent AI for deep stock analysis, ARIMA price forecasting, and RAG-enhanced insights from financial research. Your personal AI investment co-pilot.",
                "keywords": "AI stock analysis, multi-agent system, ARIMA forecast, portfolio analysis, RAG, investment research, financial AI"
            },
            "demo": {
                "page_title": "Demo - AI Stock Analysis | See Our Agent in Action",
                "page_icon": "📊",
                "meta_description": "Interactive demo of our AI stock research agent. See previews of dividend, news, and quantitative analysis for global stocks.",
                "keywords": "AI stock demo, investment analysis preview, quant analysis, dividend analysis"
            },
            "about": {
                "page_title": "About - The Technology Behind Our AI Investment Co-Pilot",
                "page_icon": "🤖",
                "meta_description": "Learn about the multi-agent architecture, ARIMA modeling, and RAG technology that power our advanced AI stock analysis platform.",
                "keywords": "AI investment technology, LangGraph, financial modeling, RAG architecture"
            }

        }

        config = configs.get(page_type, configs["landing"])

        st.set_page_config(
            page_title=config["page_title"],
            page_icon=config["page_icon"],
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown("""
        <script>
        // Detect dark mode preference and set data attribute
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.setAttribute('data-theme', 'dark');
        }

        // Listen for theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
            document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        });
        </script>
        """, unsafe_allow_html=True)

        # Add comprehensive meta tags
        # Add comprehensive meta tags
        st.markdown(f"""
        <meta name="description" content="{config['meta_description']}">
        <meta name="keywords" content="{config['keywords']}">
        <meta name="robots" content="index, follow">
        <meta name="author" content="Rajesh Ranjan">

        <!-- Open Graph / Facebook -->
        <meta property="og:type" content="website">
        <meta property="og:url" content="https://rajeshranjan.click/">
        <meta property="og:title" content="{config['page_title']}">
        <meta property="og:description" content="{config['meta_description']}">
        <meta property="og:image" content="https://rajeshranjan.click/og-image.png">

        <!-- Twitter -->
        <meta property="twitter:card" content="summary_large_image">
        <meta property="twitter:url" content="https://rajeshranjan.click/">
        <meta property="twitter:title" content="{config['page_title']}">
        <meta property="twitter:description" content="{config['meta_description']}">

        <!-- Structured Data for Global Application -->
        <script type="application/ld+json">
        {{
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            "name": "AI Investment Co-Pilot",
            "description": "{config['meta_description']}",
            "url": "https://rajeshranjan.click",
            "applicationCategory": "FinanceTool",
            "operatingSystem": "Web Browser",
            "offers": {{
                "@type": "Offer",
                "price": "0"
            }},
            "areaServed": {{
                "@type": "Place",
                "name": "Global"
            }}
        }}
        </script>

        <!-- Modern CSS for enhanced UI with Dark Mode Support -->
        <style>
            /* CSS Variables for Light/Dark Mode */
            :root {{
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-gradient: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
                --text-primary: #000000;
                --text-secondary: #666666;
                --card-bg: #ffffff;
                --card-border: rgba(0, 0, 0, 0.1);
                --card-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                --nav-bg: rgba(255, 255, 255, 0.95);
                --nav-text: #1a2a6c;
                --nav-hover-bg: #1a2a6c;
                --nav-hover-text: #ffffff;
            }}

            @media (prefers-color-scheme: dark) {{
                :root {{
                    --bg-primary: #0f1116;
                    --bg-secondary: #1e1e1e;
                    --bg-gradient: linear-gradient(135deg, #2d3a8c, #c23f3f, #fdcb4d);
                    --text-primary: #ffffff;
                    --text-secondary: #cccccc;
                    --card-bg: #1e1e1e;
                    --card-border: rgba(255, 255, 255, 0.1);
                    --card-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                    --nav-bg: rgba(30, 30, 30, 0.95);
                    --nav-text: #90caf9;
                    --nav-hover-bg: #90caf9;
                    --nav-hover-text: #1a2a6c;
                }}
            }}





            /* Fix the main content area to remove top padding */
            .main .block-container {{
                padding-top: 0rem;
                padding-bottom: 0rem;
                background-color: var(--bg-primary);
                color: var(--text-primary);
            }}





            /* Enhanced landing page styles */
            .hero-section {{
                background: var(--bg-gradient);
                background-size: 400% 400%;
                animation: gradientBG 15s ease infinite;
                padding: 4rem 2rem;
                border-radius: 16px;
                margin-bottom: 3rem;
                color: white;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}

            @keyframes gradientBG {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}

            .hero-section::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100" opacity="0.05"><path d="M0,0 L100,100 M100,0 L0,100" stroke="white" stroke-width="1"/><path d="M50,0 L50,100 M0,50 L100,50" stroke="white" stroke-width="0.5"/></svg>');
                opacity: 0.1;
            }}

            .hero-title {{
                font-size: 3.5rem;
                font-weight: 800;
                margin-bottom: 1rem;
                animation: fadeInUp 1s ease;
                color: white;
            }}

            .hero-subtitle {{
                font-size: 1.5rem;
                font-weight: 300;
                margin-bottom: 2rem;
                animation: fadeInUp 1.2s ease;
                color: white;
            }}

            .cta-button {{
                background: linear-gradient(45deg, #FFD700, #FFA500);
                color: #1a2a6c;
                border: none;
                padding: 1rem 2.5rem;
                font-size: 1.2rem;
                font-weight: 600;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
                animation: fadeInUp 1.4s ease;
            }}

            .cta-button:hover {{
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(255, 215, 0, 0.6);
            }}

            .feature-section {{
                padding: 3rem 0;
            }}

            .feature-card {{
                background: var(--card-bg);
                border-radius: 12px;
                padding: 2rem;
                box-shadow: var(--card-shadow);
                transition: all 0.3s ease;
                height: 100%;
                border-left: 4px solid #1f77b4;
                color: var(--text-primary);
            }}

            .feature-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
            }}

            .feature-icon {{
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }}

            .trust-section {{
                background: var(--bg-secondary);
                padding: 3rem 2rem;
                border-radius: 16px;
                margin: 3rem 0;
                text-align: center;
                color: var(--text-primary);
            }}

            .how-it-works {{
                padding: 3rem 0;
            }}

            .step-card {{
                background: var(--card-bg);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                box-shadow: var(--card-shadow);
                height: 100%;
                color: var(--text-primary);
            }}

            .step-number {{
                width: 50px;
                height: 50px;
                background: #1f77b4;
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 1.2rem;
                margin: 0 auto 1rem;
            }}

            .sticky-nav {{
                position: sticky;
                top: 0;
                background: var(--nav-bg);
                backdrop-filter: blur(10px);
                z-index: 1000;
                padding: 1rem 0;
                margin-bottom: 2rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                border-radius: 0 0 12px 12px;
            }}

            .nav-container {{
                display: flex;
                justify-content: center;
                gap: 2rem;
            }}

            .nav-link {{
                color: var(--nav-text);
                text-decoration: none;
                font-weight: 500;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                transition: all 0.3s ease;
            }}

            .nav-link:hover {{
                background: var(--nav-hover-bg);
                color: var(--nav-hover-text);
            }}

            @keyframes fadeInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            .animated-element {{
                opacity: 0;
                animation: fadeInUp 0.8s ease forwards;
            }}

            .delay-1 {{ animation-delay: 0.2s; }}
            .delay-2 {{ animation-delay: 0.4s; }}
            .delay-3 {{ animation-delay: 0.6s; }}

            /* Additional dark mode specific adjustments */
            .stSelectbox, .stButton, .stTextInput, .stTextArea {{
                color: var(--text-primary) !important;
            }}

            .stSelectbox > div, .stTextInput > div, .stTextArea > div {{
                background-color: var(--card-bg) !important;
                border-color: var(--card-border) !important;
                color: var(--text-primary) !important;
            }}

            .stButton > button {{
                background-color: var(--nav-hover-bg) !important;
                color: var(--nav-hover-text) !important;
                border: 1px solid var(--card-border) !important;
            }}

            @media (max-width: 768px) {{
                .hero-title {{
                    font-size: 2.5rem;
                }}

                .hero-subtitle {{
                    font-size: 1.2rem;
                }}

                .nav-container {{
                    flex-direction: column;
                    gap: 0.5rem;
                }}

                section[data-testid="stSidebar"] ~ div button[title="Show sidebar"] {{
                    top: 0.5rem;
                    left: 0.5rem;
                }}
            }}
        </style>
        """, unsafe_allow_html=True)


def create_how_it_works_page():
    """Create a detailed 'How It Works' page explaining the system architecture"""
    seo = SEOEntryPoints()
    seo.setup_page_config("about")

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title animated-element">How Our AI Co-Pilot Works 🔧</h1>
        <p class="hero-subtitle animated-element delay-1">Understanding the technology behind your investment insights</p>
    </div>
    """, unsafe_allow_html=True)

    # DeepAgent Research Section
    st.markdown("""
    <div class="tab-card tab1">
        <h2><i class="fas fa-robot"></i> DeepAgent Research</h2>
        <p class="lead">Advanced multi-agent AI system for comprehensive financial analysis</p>
    """, unsafe_allow_html=True)

    # DeepAgent features using Streamlit components
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Technical Specifications")
        st.markdown("""
        - **Enabled via**: `DEEPAGENT_ENABLED` environment variable
        - **Memory**: Conversational memory persists only for session duration
        - **Architecture**: Uses LangGraph's stateful architecture for agent coordination
        - **Human Interaction**: Human-in-the-Loop not available in DeepAgent mode
        """)

    with col2:
        st.markdown("### Capabilities")
        st.markdown("""
        - **Multi-Agent System**: Specialized AI agents working collaboratively
        - **RAG Integration**: Combines Retrieval-Augmented Generation with memory
        - **Configurable**: Can be enabled/disabled via environment settings
        - **Fallback**: Automatically switches to standard analysis when needed
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # Express Analysis Section
    st.markdown("""
    <div class="tab-card tab2">
        <h2><i class="fas fa-bolt"></i> Express Analysis</h2>
        <p class="lead">Rapid financial assessment with human oversight</p>
    """, unsafe_allow_html=True)

    # Express Analysis features using Streamlit components
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Technical Specifications")
        st.markdown("""
        - **Processing**: Designed for rapid analysis without memory persistence
        - **Human Feedback**: Up to two revision cycles with Human-in-the-Loop
        - **RAG**: Evidence-based insights with Retrieval-Augmented Generation
        - **Workflow**: Uses LangGraph without multi-agent complexity
        """)

    with col2:
        st.markdown("### Capabilities")
        st.markdown("""
        - **Speed**: Streamlined processing for quick results
        - **Oversight**: Human validation and refinement capabilities
        - **Focused**: Targeted insights for specific analysis needs
        - **Flexible**: Adaptable to different research requirements
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # System Architecture Overview
    st.markdown("""
    <div class="architecture-diagram">
        <h2><i class="fas fa-project-diagram"></i> System Architecture Overview</h2>
    """, unsafe_allow_html=True)

    st.markdown("### Workflow Process")

    # Architecture steps
    steps = [
        {"icon": "👤", "title": "User Query", "desc": "User submits financial research request"},
        {"icon": "⚙️", "title": "System Check", "desc": "Platform checks DeepAgent availability"},
        {"icon": "🔄", "title": "Routing", "desc": "Directs to appropriate analysis pipeline"}
    ]

    # Display steps horizontally
    cols = st.columns(3)
    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(f"<div style='text-align: center;'><span style='font-size: 2rem;'>{step['icon']}</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center;'>{step['title']}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{step['desc']}</p>", unsafe_allow_html=True)

    # Architecture branches
    st.markdown("### Analysis Paths")
    branch_cols = st.columns(2)

    with branch_cols[0]:
        st.markdown("#### DeepAgent Path")
        st.markdown("""
        - Multi-agent analysis with memory
        - Advanced RAG capabilities
        - ARIMA forecasting
        - Context-aware dialogues
        """)

    with branch_cols[1]:
        st.markdown("#### Express Path")
        st.markdown("""
        - Rapid analysis with HIL
        - Streamlined processing
        - Human feedback integration
        - Focused insights
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # Technical Implementation Details
    st.markdown("""
    <div class="tab-card">
        <h2><i class="fas fa-cogs"></i> Technical Implementation Details</h2>
    """, unsafe_allow_html=True)

    # Technical details in tabs
    tab1, tab2, tab3 = st.tabs(["DeepAgent Research", "Express Analysis", "Common Infrastructure"])

    with tab1:
        st.markdown("### DeepAgent Technical Details")
        st.markdown("""
        - **Activation**: Controlled by `DEEPAGENT_ENABLED` environment variable
        - **Memory**: Session-based conversational memory
        - **Architecture**: LangGraph stateful architecture for agent coordination
        - **Human Interaction**: No Human-in-the-Loop in this mode
        - **Fallback**: Automatic switch to standard analysis when needed
        """)

    with tab2:
        st.markdown("### Express Analysis Technical Details")
        st.markdown("""
        - **Processing**: Optimized for speed without memory persistence
        - **Human Feedback**: Two-round revision system with Human-in-the-Loop
        - **RAG**: Evidence-based insights with retrieval capabilities
        - **Workflow**: LangGraph for streamlined workflow management
        """)

    with tab3:
        st.markdown("### Common Infrastructure")
        st.markdown("""
        - **MCP Server**: Model Context Protocol for model interoperability
        - **Cloud Deployment**: AWS ECS Fargate for scalability
        - **Security**: Comprehensive financial data protection
        - **Caching**: Intelligent data caching for optimal performance
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # Add CSS for the How It Works page
    st.markdown("""
    <style>
        .tab-card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: var(--card-shadow);
            margin-bottom: 2rem;
            transition: all 0.3s ease;
            border-left: 4px solid var(--secondary-color);
            color: var(--text-primary);
        }

        .tab-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }

        .tab-card.tab1 {
            border-left-color: #3498db;
        }

        .tab-card.tab2 {
            border-left-color: #e74c3c;
        }

        .architecture-diagram {
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            margin: 2rem 0;
            color: var(--text-primary);
        }

        @media (max-width: 768px) {
            .tab-card {
                padding: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # CTA Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a2a6c, #b21f1f); padding: 4rem 2rem; border-radius: 16px; color: white; text-align: center; margin: 3rem 0;">
        <h2 style="margin-bottom: 1.5rem;">Ready to Experience Our Technology?</h2>
        <p style="margin-bottom: 2rem; font-size: 1.2rem;">Try our AI Co-Pilot with both DeepAgent and Express Analysis capabilities</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("🚀 Try AI Co-Pilot Now", type="primary", use_container_width=True, key="how_it_works_cta"):
            st.session_state.show_login_modal = True
            # st.rerun()
    if st.session_state.get("show_login_modal", False):
        show_modal_spinner("Loading login page...")
        st.session_state.update({
            'show_login_modal': False,
            'show_login': True
        })
        # This single rerun will take the user to the login page
        st.rerun()


def create_landing_page():
    """Create SEO-optimized landing page with enhanced UI"""
    seo = SEOEntryPoints()
    seo.setup_page_config("landing")

    # Sticky navigation
    st.markdown("""
    <div class="sticky-nav">
        <div class="nav-container">
            <a href="#features" class="nav-link">Features</a>
            <a href="#how-it-works" class="nav-link">How It Works</a>
            <a href="#stocks" class="nav-link">Stocks</a>
            <a href="#try-now" class="nav-link">Try Now</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero section with animation (without the redundant button)
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title animated-element">Your AI Investment Co-Pilot 🚀</h1>
        <p class="hero-subtitle animated-element delay-1">Smarter stock research. Faster insights. Make confident decisions.</p>
        <p class="animated-element delay-2" style="font-size: 1.1rem; margin-top: 2rem;">
            Scroll down to explore powerful features and try our AI analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key features section - Moved up to replace the removed section
    st.markdown("""
    <div class="feature-section" id="features">
        <h2 style="text-align: center; margin-bottom: 2.5rem;">Powerful Features for Smarter Investing</h2>
        <div class="row" style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""


        <div class="feature-card animated-element">
            <div class="feature-icon">🧠</div>
            <h3>DeepAgent Analysis</h3>
            <p><strong>Multi-Agent System:</strong> Our specialized AI agents work in concert to dissect complex financial data, delivering institutional-grade research previously unavailable to retail investors.</p>
            <p><strong>ARIMA Forecasting:</strong> Leverage advanced statistical modeling to anticipate price movements with quantifiable confidence intervals, giving you an edge in volatile markets.</p>
            <p><strong>Actionable Insights:</strong> Receive clear, data-driven Buy/Sell/Hold recommendations with supporting rationale to inform your investment decisions with confidence.</p>
            <p><strong>Conversation Memory:</strong> Experience truly personalized analysis that evolves with your inquiry patterns, maintaining context across sessions for deeper investigative continuity.</p>
            <p><strong>Human-in-the-Loop Validation:</strong> Maintain strategic oversight with the ability to guide, correct, and refine AI analysis, combining machine speed with human wisdom.</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
       <div class="feature-card animated-element delay-1">
        <div class="feature-icon">📈</div>
        <h3>Comprehensive Research Suite</h3>
        <p><strong>Holistic Analysis:</strong> Go beyond price action with integrated assessment of dividend sustainability, news sentiment impact, and 360-degree quantitative metrics.</p>
        <p><strong>Portfolio Intelligence:</strong> Analyze correlation and concentration risks across up to 10 positions simultaneously, optimizing your portfolio's risk-return profile.</p>
        <p><strong>Custom Timeframes:</strong> Align analytical periods with your specific investment horizon, from day trading to long-term value investing strategies.</p>
        <p><strong>Express Analysis with Human Oversight:</strong> Get rapid insights without sacrificing accuracy, with optional expert intervention points for critical decision validation.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""


        <div class="feature-card animated-element delay-2">
        <div class="feature-icon">🔬</div>
        <h3>Cutting-Edge Technology</h3>
        <p><strong>RAG-Enhanced:</strong> Augment AI reasoning with insights extracted from thousands of institutional financial papers and SEC filings, creating analysis depth rivaling Wall Street research desks.</p>
        <p><strong>PDF Export:</strong> Generate professional-grade research reports ready for compliance review, investment committee presentations, or personal archives.</p>
        <p><strong>Interactive & Reactive:</strong> Experience seamless, dynamic analysis powered by LangGraph architecture that responds to market conditions in near real-time.</p>
        <p><strong>Intelligent Caching:</strong> Balance speed and accuracy with our sophisticated caching system that stores static financial data optimally to deliver insights faster than competitors.</p>
        <p><strong>MCP Server Integration:</strong> Leverage the latest in Model Context Protocol technology for unprecedented interoperability between analytical models and data sources, future-proofing your analytical capabilities.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # How it works section
    st.markdown("""
    <div class="how-it-works" id="how-it-works">
        <h2 style="text-align: center; margin-bottom: 2.5rem;">How It Works in 3 Simple Steps</h2>
        <div class="row" style="display: flex; gap: 1.5rem; flex-wrap: wrap; justify-content: center;">
            <div style="flex: 1; min-width: 250px;">
                <div class="step-card animated-element">
                    <div class="step-number">1</div>
                    <h3>Enter Stock Ticker</h3>
                    <p>Select from thousands of global stocks or input your own ticker symbol</p>
                </div>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <div class="step-card animated-element delay-1">
                    <div class="step-number">2</div>
                    <h3>Get AI-Powered Analysis</h3>
                    <p>Our multi-agent system performs deep fundamental, technical, and sentiment analysis</p>
                </div>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <div class="step-card animated-element delay-2">
                    <div class="step-number">3</div>
                    <h3>Make Smarter Decisions</h3>
                    <p>Receive actionable insights and forecasts to guide your investment strategy</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample stocks section for SEO content
    st.markdown("""
    <div id="stocks" style="padding: 3rem 0;">
        <h2 style="text-align: center; margin-bottom: 2rem;">🏦 Analyze Thousands of Global Stocks</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center;">
    """, unsafe_allow_html=True)

    stock_cols = st.columns(4)
    for i, stock in enumerate(seo.sample_stocks):
        with stock_cols[i % 4]:
            st.markdown(f"""
            <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px; box-shadow: var(--card-shadow); text-align: center; color: var(--text-primary);">
                <strong>{stock}</strong>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # CTA Section
    st.markdown("""
    <div id="try-now" style="background: linear-gradient(135deg, #1a2a6c, #b21f1f); padding: 4rem 2rem; border-radius: 16px; color: white; text-align: center; margin: 3rem 0;">
        <h2 style="margin-bottom: 1.5rem;">💡 Unlock Institutional-Grade Research Today</h2>
        <p style="margin-bottom: 2rem; font-size: 1.2rem;">Go beyond basic data. Our AI Co-Pilot provides the depth and intelligence previously reserved for financial institutions.</p>
        <p style="margin-bottom: 2rem; font-size: 1.2rem;">Start making data-driven decisions with confidence.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create a container for the CTA button to center it
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    #     if st.button("🔐 Try Free Research Now", type="primary", use_container_width=True, key="final_cta"):
    #         st.session_state.show_login = True
    #         st.rerun()

    # col1, col2, col3 = st.columns([1.5, 1, 1.5])
    # with col2:
    #     if st.button("🔐 Try Free Research Now", type="primary", use_container_width=True, key="final_cta"):
    #         st.session_state.show_login_modal = True
    #         st.rerun()
    #
    # if st.session_state.get("show_login_modal", False):
    #     show_modal_spinner("Loading login page...")
    #     st.session_state.show_login_modal = False
    #     st.session_state.show_login = True
    #     st.rerun()
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("🔐 Try Free Research Now", type="primary", use_container_width=True, key="final_cta"):
            # Combine all state changes in a single update
            st.session_state.update({
                'show_login_modal': False,
                'show_login': True,
                # 'login_transition_started': True  # Optional: track transition state
            })

            # Use your existing spinner component
            show_modal_spinner("Loading login page...")
            st.rerun()


def create_demo_page():
    """Create enhanced demo page with interactive elements and expandable responses"""
    seo = SEOEntryPoints()
    seo.setup_page_config("demo")

    # Initialize session state for demo responses
    if 'expanded_response' not in st.session_state:
        st.session_state.expanded_response = None

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title animated-element">See the Co-Pilot in Action 🎬</h1>
        <p class="hero-subtitle animated-element delay-1">Explore how AI transforms stock research into instant insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Step-by-Step Walkthrough
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <h2>How It Works in 3 Simple Steps</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="step-card animated-element">
            <div class="step-number">1</div>
            <h3>Ask a Question</h3>
            <p>Enter any stock research question or select from our predefined queries</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="step-card animated-element delay-1">
            <div class="step-number">2</div>
            <h3>AI Analysis</h3>
            <p>Our multi-agent system scans fundamentals, news, and market data</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="step-card animated-element delay-2">
            <div class="step-number">3</div>
            <h3>Get Insights</h3>
            <p>Receive comprehensive analysis with clear recommendations</p>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Demo Section
    st.markdown("""
    <div style="background: var(--bg-secondary); padding: 3rem 2rem; border-radius: 16px; margin: 3rem 0;">
        <h2 style="text-align: center; margin-bottom: 2rem;">Try Sample Queries</h2>
        <p style="text-align: center; color: var(--text-secondary); margin-bottom: 2rem;">Click any query to see sample AI responses</p>
    </div>
    """, unsafe_allow_html=True)

    # Create interactive query buttons using Streamlit
    query_col1, query_col2, query_col3, query_col4 = st.columns(4)

    with query_col1:
        if st.button("Forecast Tesla\nnext 5 years", use_container_width=True, key="tesla_forecast"):
            st.session_state.expanded_response = "tesla_forecast"
            st.rerun()

    with query_col2:
        if st.button("Compare Apple vs\nMicrosoft dividends", use_container_width=True, key="dividend_comparison"):
            st.session_state.expanded_response = "dividend_comparison"
            st.rerun()

    with query_col3:
        if st.button("NVIDIA technical\nanalysis", use_container_width=True, key="nvidia_analysis"):
            st.session_state.expanded_response = "nvidia_analysis"
            st.rerun()

    with query_col4:
        if st.button("Amazon news\nsentiment", use_container_width=True, key="amazon_sentiment"):
            st.session_state.expanded_response = "amazon_sentiment"
            st.rerun()

    # Show expandable sample responses
    if st.session_state.expanded_response:
        st.markdown("---")
        st.markdown("### 🤖 AI Sample Response")

        if st.session_state.expanded_response == "tesla_forecast":
            with st.expander("Tesla 5-Year ARIMA Forecast Analysis", expanded=True):
                st.markdown("""
                **📈 Tesla Inc. (TSLA) 5-Year ARIMA Forecast Analysis**

                **Forecast Summary:**
                • **Model:** ARIMA(5,1,0) fitted to the last 250 trading days of TSLA (through 2025-09-02)
                • **Drift:** None (d=1), no MA component (q=0), five AR lags (p=5)
                • **In‐sample AIC:** 2004.31, σ²≈174.7
                • **Mean Reversion:** Strong tendency to revert to ~$331.91

                **Key Price Targets:**
                • **1-Year Ahead (≈252 trading days):**
                  - Point forecast: $331.91
                  - 95% CI (Day 252): [$331.91, $331.91]

                • **3-Year Ahead (≈756 trading days):**
                  - Point forecast: $331.91  
                  - 95% CI (Day 756): [$331.91, $331.91]

                • **5-Year Ahead (≈1,260 trading days):**
                  - Point forecast: $331.91
                  - 95% CI (Day 1,260): [$331.91, $331.91]

                **Methodology & Assumptions:**
                • Univariate ARIMA model on daily closing prices, differenced once for stationarity
                • No exogenous regressors - forecast reflects only historical price autocorrelation
                • Multi-year horizons converge to unconditional mean, producing flat forecast

                **Limitations & Risks:**
                • ⚠️ Tesla's business fundamentals (new products, regulations, competition) not captured
                • ⚠️ EV market volatility, interest-rate cycles, sentiment shifts can cause large deviations  
                • ⚠️ Narrow confidence intervals are model artifacts, not realistic certainty measures

                **Bottom Line:**
                • Under pure ARIMA historical analysis, TSLA reverts to ~$332 over 5 years
                • For forward-looking forecasts, incorporate:
                  - Fundamental drivers & scenario analysis
                  - Flexible time-series methods (state-space, regime-switching)
                  - Macroeconomic variables

                **Recommendation:** 🟡 **HOLD** (Seek Additional Analysis)
                *ARIMA alone suggests mean reversion, but fundamental catalysts may drive different outcomes*
                """)

        elif st.session_state.expanded_response == "dividend_comparison":
            with st.expander("Apple vs Microsoft Dividend Analysis", expanded=True):
                st.markdown("""
                **📊 Apple (AAPL) vs. Microsoft (MSFT) Dividend Comparison**

                **Dividend Level & Yield**
                - **Apple (AAPL):**
                  - Annualized dividend rate: \$1.04 per share
                  - Dividend yield (forward): 0.45%
                  - Five-year average yield: 0.54%
                - **Microsoft (MSFT):**
                  - Annualized dividend rate: \$3.32 per share
                  - Dividend yield (forward): 0.66%
                  - Five-year average yield: 0.82%

                **Payout Ratio**
                - AAPL payout ratio (trailing): 15.3% of net income
                - MSFT payout ratio (trailing): 23.8% of net income

                **Dividend Growth**
                - AAPL: Increased quarterly payout from \$0.22 (early 2022) to \$0.26 (mid 2025) – ≈18% total boost (~6% CAGR)
                - MSFT: Increased quarterly payout from \$0.62 (late 2021) to \$0.83 (mid 2025) – ≈34% total boost (~9% CAGR)

                **Valuation Context**
                - AAPL forward P/E ≈ 27.6
                - MSFT forward P/E ≈ 33.8

                **Upcoming Key Dates**
                - AAPL ex-dividend date: August 11, 2025
                - MSFT ex-dividend date: August 21, 2025

                **Interpretation & Takeaway**
                - **Income Focus:** MSFT offers a higher current yield (0.66% vs. 0.45%) and faster dividend growth (≈9% vs. ≈6% annualized). Its payout ratio is conservative, allowing room for future increases.
                - **Total-Return/Growth Blend:** AAPL’s lower payout ratio (≈15%) means more earnings are reinvested, supporting buybacks and growth. Its lower valuation may appeal to those seeking capital appreciation.
                - **Risk & Sustainability:** Both have strong cash flows and low payout ratios, making dividends highly sustainable. MSFT’s higher leverage does not materially threaten its payout.
                - **Our View:** MSFT is slightly more attractive for income. AAPL suits those wanting a mix of income and growth. Both are industry leaders with well-covered dividends and strong analyst sentiment.
                """, unsafe_allow_html=True)

        elif st.session_state.expanded_response == "nvidia_analysis":
            with st.expander("NVIDIA overall Analysis", expanded=True):
                st.markdown("""
            **📊 NVIDIA (NVDA) – 6-Month Technical Analysis Overview**

            **Trend Analysis**
            - Uptrend with periodic corrections: NVDA rose from ~$110 (March) to ~$180 (August), maintaining higher highs/lows since mid-April.
            - Recent pullback: August high near $184 to ~$170 (Sept, ~7.6% retracement). Still above 50-day MA (~$162), long-term bullish bias intact.

            **Moving Averages**
            - 20-day SMA: ~$179.4 (price at $171–174, slightly below short MA, consolidation).
            - 50-day SMA: ~$171.1 (price hovers above/below, equilibrium zone).
            - 200-day SMA: ~$145 (extrapolated, price well above, bullish longer-term).

            **RSI (14-Day)**
            - Late-August oversold spike to ~30, recent rebound to ~45.
            - Current RSI ~40–45: neutral momentum, not overbought, room to rally.

            **MACD**
            - MACD line positive (~+1.3) over signal, but momentum slowing from July peak (~7.0).
            - Histogram contracting: momentum slowing, still favors buyers.

            **Bollinger Bands (20,2)**
            - Bands widened in July/Aug rally (~146/180), now contracting.
            - Price testing lower band (~172): potential support/entry zone.

            **Volume & OBV**
            - Volume spikes on sell-offs, lighter on rallies—watch for conviction.
            - OBV peaked mid-July, slight decline—monitor for renewed accumulation.

            **Chart Patterns**
            - Bull Flag: June–July consolidation preceded July rally.
            - Symmetrical Triangle: July–August, breakdown into $170 area. Breakout above $180 or below $170 sets next leg.

            **Support & Resistance**
            - Support: $170 (50-day MA), $162 (May swing low).
            - Resistance: $180–183 (recent highs), $185–187 (upper BB).

            **Key Levels**
            - Near-term support: $170–172 (lower BB, 50-day MA).
            - Near-term resistance: $180–183.
            - Stop-loss: Below $168 (break below 50-day MA).
            - Upside target: $185 initial, then $200 psychological.

            **Technical Outlook & Recommendation**
            - Neutral-bullish: Pullback within healthy uptrend; moving averages provide support.
            - **Strategy:** Accumulate on dips near $170 with stop at $168; look for volume confirmation and RSI rebound.
            - Break above $183 on rising volume: add to position, target $200. Close below $168: caution/exit.

            **Summary**
            - NVDA’s uptrend remains intact but is consolidating. Indicators show neutral momentum, no overbought condition.
            - Defense of $170–172 zone could offer low-risk entry; failure may signal deeper correction.
            - Monitor volume and RSI for confirmation of directional bias.
            """, unsafe_allow_html=True)

        elif st.session_state.expanded_response == "amazon_sentiment":
            with st.expander("Amazon News Sentiment Analysis", expanded=True):
                st.markdown("""
                **📰 Amazon (AMZN) – Latest News Sentiment Analysis**

                | Date       | Source         | Headline                                                        | Sentiment         | Impact Summary                                                                 |
                |------------|----------------|-----------------------------------------------------------------|-------------------|--------------------------------------------------------------------------------|
                | 06-17-2025 | Reuters        | Amazon announces four-day Prime Day discount event               | Positive          | Extended Prime Day boosts sales, member engagement, and topline growth.         |
                | 07-07-2025 | Reuters/YouTube| Amazon Prime Day to lift online sales to 23.8 billion            | Positive          | E-commerce surge, pricing power, near-term catalyst for shares.                |
                | 09-02-2025 | Reuters        | Amazon U.S. Prime sign-ups slow despite expanded Prime Day push  | Negative          | Sign-up miss signals slowing engagement, may pressure recurring revenue.        |
                | 07-11-2025 | TipRanks       | Amazon stock fails to deliver as Prime Day data disappoints      | Negative          | Shares dip, consensus still strong, but near-term volatility expected.          |
                | 09-01-2025 | MarketScreener | Amazon to invest 4.4 billion in New Zealand data centers         | Positive          | Major AWS capex, supports cloud growth, offsets retail margin pressure.         |
                | 06-23-2025 | Yahoo Finance  | Model aircraft club goes to war with Amazon over drone deliveries| Neutral/Negative  | Regulatory friction for Prime Air, modest impact on core operations.            |

                **Overall Sentiment Summary**
                - Amazon continues to innovate in retail and cloud (Prime Day, AWS investment).
                - Early signs of softening engagement (Prime sign-up shortfall) and regulatory push-back (drone delivery).
                - Analyst consensus constructive, but share performance may be volatile around event-driven data.

                **Actionable Insights**
                - Watch official Prime Day sign-up and GMV data—further misses could trigger a pullback.
                - AWS expansion is a durable growth driver; positive for long-term exposure.
                - Regulatory issues around drone delivery are localized; unlikely to derail logistics rollout.
                - Consider using near-term volatility around Prime metrics to add exposure, aiming to hold through next earnings.
                """, unsafe_allow_html=True)

        # Add a button to collapse all responses
        if st.button("Collapse Response", key="collapse_response"):
            st.session_state.expanded_response = None
            st.rerun()

    # Features Showcase
    st.markdown("""
    <div style="margin: 3rem 0;">
        <h2 style="text-align: center; margin-bottom: 2rem;">Powerful Analysis Capabilities</h2>
    </div>
    """, unsafe_allow_html=True)

    # Create analysis capability cards using Streamlit columns
    analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)

    # Create analysis capability cards using Streamlit columns
    analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)

    with analysis_col1:
        st.markdown("""
        <div style="background: var(--card-bg); border-radius: 12px; padding: 1.5rem; box-shadow: var(--card-shadow); height: 100%; color: var(--text-primary);">
            <h4 style="color: var(--nav-text); margin-bottom: 1rem;">📈 Dividend Analysis</h4>
            <p style="font-size: 0.9rem;">Comprehensive dividend yield, history, and sustainability analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with analysis_col2:
        st.markdown("""
        <div style="background: var(--card-bg); border-radius: 12px; padding: 1.5rem; box-shadow: var(--card-shadow); height: 100%; color: var(--text-primary);">
            <h4 style="color: var(--nav-text); margin-bottom: 1rem;">📰 News Sentiment</h4>
            <p style="font-size: 0.9rem;">Real-time news analysis with sentiment scoring and impact assessment</p>
        </div>
        """, unsafe_allow_html=True)

    with analysis_col3:
        st.markdown("""
        <div style="background: var(--card-bg); border-radius: 12px; padding: 1.5rem; box-shadow: var(--card-shadow); height: 100%; color: var(--text-primary);">
            <h4 style="color: var(--nav-text); margin-bottom: 1rem;">🔢 Quantitative Analysis</h4>
            <p style="font-size: 0.9rem;">Deep financial metrics analysis with industry benchmarking</p>
        </div>
        """, unsafe_allow_html=True)

    with analysis_col4:
        st.markdown("""
        <div style="background: var(--card-bg); border-radius: 12px; padding: 1.5rem; box-shadow: var(--card-shadow); height: 100%; color: var(--text-primary);">
            <h4 style="color: var(--nav-text); margin-bottom: 1rem;">🔮 Price Forecasting</h4>
            <p style="font-size: 0.9rem;">ARIMA-based price predictions with confidence intervals</p>
        </div>
        """, unsafe_allow_html=True)

    # Social Proof
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 3rem 2rem; border-radius: 16px; margin: 3rem 0;">
        <h2 style="text-align: center; margin-bottom: 2rem;">Loved by Investors & Researchers</h2>
    </div>
    """, unsafe_allow_html=True)

    # Create testimonial cards using Streamlit columns
    testimonial_col1, testimonial_col2, testimonial_col3 = st.columns(3)

    # Create testimonial cards using Streamlit columns
    testimonial_col1, testimonial_col2, testimonial_col3 = st.columns(3)

    with testimonial_col1:
        st.markdown("""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; box-shadow: var(--card-shadow); height: 100%; color: var(--text-primary);">
            <p style="font-style: italic; color: var(--text-secondary); font-size: 0.95rem;">"This tool cut my research time by 80% while improving accuracy. Game-changer!"</p>
            <p style="font-weight: bold; color: var(--nav-text); margin-top: 1rem; font-size: 0.9rem;">- Financial Analyst</p>
        </div>
        """, unsafe_allow_html=True)

    with testimonial_col2:
        st.markdown("""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; box-shadow: var(--card-shadow); height: 100%; color: var(--text-primary);">
            <p style="font-style: italic; color: var(--text-secondary); font-size: 0.95rem;">"The multi-agent approach provides insights I wouldn't have considered on my own."</p>
            <p style="font-weight: bold; color: var(--nav-text); margin-top: 1rem; font-size: 0.9rem;">- Portfolio Manager</p>
        </div>
        """, unsafe_allow_html=True)

    with testimonial_col3:
        st.markdown("""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; box-shadow: var(--card-shadow); height: 100%; color: var(--text-primary);">
            <p style="font-style: italic; color: var(--text-secondary); font-size: 0.95rem;">"Finally, an AI tool that understands the nuances of financial analysis."</p>
            <p style="font-weight: bold; color: var(--nav-text); margin-top: 1rem; font-size: 0.9rem;">- Investment Researcher</p>
        </div>
        """, unsafe_allow_html=True)

    # Original Demo Content (Integrated with better styling)
    st.markdown("""
    <div style="margin: 4rem 0;">
        <h2 style="text-align: center; margin-bottom: 2rem;">📊 Interactive Demo</h2>
        <p style="text-align: center; color: var(--text-secondary); margin-bottom: 2rem;">
            Experience a glimpse of the AI Co-Pilot's capabilities with this interactive demo
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Demo stock selector
    demo_stock = st.selectbox(
        "Select a stock for demo analysis:",
        ["Apple (AAPL)", "Microsoft (MSFT)", "Tesla (TSLA)", "DBS Group (D05.SI)"],
        key="demo_stock_selector"
    )

    if demo_stock:
        stock_code = demo_stock.split("(")[1].replace(")", "")

        # Sample analysis sections with enhanced styling
        # Sample analysis sections with enhanced styling
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 2rem; border-radius: 12px; box-shadow: var(--card-shadow); margin: 2rem 0; color: var(--text-primary);">
            <h3 style="color: var(--nav-text); margin-bottom: 1.5rem;">Analysis for {demo_stock}</h3>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📋 Overview", "📈 Quant Preview", "🔍 News & Dividends"])

        with tab1:
            st.markdown(f"""
            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 8px; color: var(--text-primary);">
                <h4>Company Snapshot</h4>
                <p><strong>Stock Code:</strong> {stock_code}</p>
                <p><strong>Exchange:</strong> NASDAQ (Sample)</p>
                <p><strong>Sector:</strong> Technology (Sample)</p>
                <p><strong>Market Cap:</strong> USD 2.8T (Sample)</p>
                <p style="font-style: italic; color: var(--text-secondary); margin-top: 1rem;">
                    This is a sample report. Login to access the <strong>Deepagent</strong> with 
                    <strong>ARIMA price forecasting</strong> and multi-agent task analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("""
            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 8px; color: var(--text-primary);">
                <h4>Quantitative Analysis Preview</h4>
                <ul>
                    <li><strong>Valuation:</strong> P/E Ratio, P/B Ratio</li>
                    <li><strong>Profitability:</strong> ROE, ROA</li>
                    <li><strong>Financial Health:</strong> Debt-to-Equity</li>
                    <li><strong>Efficiency:</strong> Asset Turnover</li>
                </ul>
                <p style="font-style: italic; color: var(--text-secondary); margin-top: 1rem;">
                    Full version includes <strong>RAG-enhanced insights</strong> from financial research papers 
                    and portfolio analysis for up to 10 stocks.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("""
            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 8px; color: var(--text-primary);">
                <h4>News & Dividend Preview</h4>
                <ul>
                    <li><strong>Sentiment Analysis:</strong> Gauges market mood from latest news</li>
                    <li><strong>Dividend Yield:</strong> Income potential analysis</li>
                    <li><strong>Payout Ratio:</strong> Sustainability of dividend payments</li>
                    <li><strong>Dividend History:</strong> Tracks past performance</li>
                </ul>
                <p style="font-style: italic; color: var(--text-secondary); margin-top: 1rem;">
                    Login to export full, detailed research reports as a <strong>PDF</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Final CTA Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a2a6c, #b21f1f); padding: 3rem 2rem; border-radius: 16px; color: white; text-align: center; margin: 3rem 0;">
        <h2 style="margin-bottom: 1.5rem;">Ready to Unlock Full AI Capabilities?</h2>
        <p style="margin-bottom: 2rem; font-size: 1.1rem;">
            This demo shows just a fraction of what our AI Co-Pilot can do. 
            Experience the full power of multi-agent analysis, ARIMA forecasting, and RAG-enhanced insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # col1, col2, col3 = st.columns([1.5, 1, 1.5])
    # with col2:
    #     if st.button("🔐 Unlock Full AI Co-Pilot", type="primary", use_container_width=True, key="final_demo_cta"):
    #         st.session_state.show_login_modal = True
    #         st.rerun()
    #
    #     if st.session_state.get("show_login_modal", False):
    #         show_modal_spinner("Loading login page...")
    #         st.session_state.show_login_modal = False
    #         st.session_state.show_login = True
    #         st.rerun()
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("🔐 Unlock Full AI Co-Pilot", type="primary", use_container_width=True, key="final_demo_cta"):
            # Combine state changes to avoid multiple reruns
            st.session_state.update({
                'show_login_modal': False,
                'show_login': True
            })

            # Use your existing spinner
            show_modal_spinner("Loading login page...")
            st.rerun()


def create_about_page():
    """Create enhanced about page with storytelling and credibility elements"""
    seo = SEOEntryPoints()
    seo.setup_page_config("about")

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title animated-element">Redefining Stock Research with AI 🚀</h1>
        <p class="hero-subtitle animated-element delay-1">Your personalized investment co-pilot powered by LLMs & multi-agent orchestration</p>
    </div>
    """, unsafe_allow_html=True)

    # Mission Storytelling
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="animated-element">
            <h2>🌟 Our Mission</h2>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                We founded this platform after experiencing firsthand how <strong>traditional stock research is slow, scattered, and overwhelmingly complex</strong>. 
                While working in finance, we spent countless hours juggling multiple data sources, struggling to connect the dots, and often missing crucial insights.
            </p>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                Our solution? An AI co-pilot that makes research <strong>faster, deeper, and more reliable</strong>. 
                We've combined cutting-edge language models with multi-agent systems to create the research assistant we always wished we had.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Old Way vs New Way comparison
        st.markdown("""
        <div class="animated-element delay-1" style="background: var(--card-bg); padding: 2rem; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h3 style="text-align: center; color: #1a2a6c; margin-bottom: 1.5rem;">Old Way vs New Way</h3>
            """, unsafe_allow_html=True)

        # Create two columns for the comparison
        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⏳</div>
                <h4 style="color: var(--text-secondary); margin-bottom: 0.5rem;">Manual Research</h4>
                <ul style="text-align: left; font-size: 0.9rem; color: var(--text-secondary);">
                    <li>Hours of data gathering</li>
                    <li>Multiple disconnected sources</li>
                    <li>Subjective analysis</li>
                    <li>Missed opportunities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with comp_col2:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                <h4 style="color: #1a2a6c; margin-bottom: 0.5rem;">AI-Powered</h4>
                <ul style="text-align: left; font-size: 0.9rem; color: #1a2a6c;">
                    <li>Instant comprehensive analysis</li>
                    <li>Unified multi-source insights</li>
                    <li>Objective AI assessment</li>
                    <li>Actionable recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Why Choose Us Section
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0;">
        <h2>Why Choose Us?</h2>
        <p style="font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 2rem;">Built with cutting-edge technology for unparalleled research experience</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-card animated-element">
            <div class="feature-icon">⚡</div>
            <h3>Speed</h3>
            <p>Get comprehensive analysis in seconds instead of hours. Process thousands of data points instantly.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card animated-element delay-1">
            <div class="feature-icon">🎯</div>
            <h3>Accuracy</h3>
            <p>AI-powered insights with institutional-grade precision. Reduced human error and bias in analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card animated-element delay-2">
            <div class="feature-icon">📊</div>
            <h3>Depth</h3>
            <p>Go beyond surface-level analysis. Deep fundamental, technical, and sentiment analysis combined.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card animated-element delay-3">
            <div class="feature-icon">🧠</div>
            <h3>Intelligence</h3>
            <p>Multi-agent system that thinks like a team of expert analysts working together on your research.</p>
        </div>
        """, unsafe_allow_html=True)

    # Trust Badges
    st.markdown("""
    <div style="background: var(--bg-secondary); padding: 3rem 2rem; border-radius: 16px; text-align: center; margin: 3rem 0;">
        <h3 style="margin-bottom: 2rem;">Backed by Advanced Technology</h3>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="background: var(--card-bg); padding: 1rem 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <span style="font-weight: bold; color: #1a2a6c;">Multi-Agent AI System</span>
            </div>
            <div style="background: var(--card-bg); padding: 1rem 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <span style="font-weight: bold; color: #1a2a6c;">MCP Framework</span>
            </div>
            <div style="background: var(--card-bg); padding: 1rem 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <span style="font-weight: bold; color: #1a2a6c;">RAG Technology</span>
            </div>
            <div style="background: var(--card-bg); padding: 1rem 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <span style="font-weight: bold; color: #1a2a6c;">ARIMA Forecasting</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a2a6c, #b21f1f); padding: 4rem 2rem; border-radius: 16px; color: white; text-align: center; margin: 3rem 0;">
        <h2 style="margin-bottom: 1.5rem;">Ready to Transform Your Research Process?</h2>
        <p style="margin-bottom: 2rem; font-size: 1.2rem;">Join thousands of investors who have already upgraded their research workflow with AI</p>
    </div>
    """, unsafe_allow_html=True)

    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    #     if st.button("⚡ Try the Co-Pilot Now", type="primary", use_container_width=True, key="about_cta"):
    #         st.session_state.show_login_modal = True
    #         st.rerun()
    #     if st.session_state.get("show_login_modal", False):
    #         show_modal_spinner("Loading login page...")
    #         st.session_state.show_login_modal = False
    #         st.session_state.show_login = True
    #         st.rerun()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚡ Try the Co-Pilot Now", type="primary", use_container_width=True, key="about_cta"):
            # Combine all state changes in a single update
            st.session_state.update({
                'show_login_modal': False,
                'show_login': True,
                'login_transition_started': True  # New flag for tracking transition
            })

            # Use your existing spinner component
            show_modal_spinner("Loading login page...")
            st.rerun()

    # Core Technology & Features Section (Original Content with Enhanced Styling)
    st.markdown("""
    <div style="margin-top: 4rem;">
        <h2 style="text-align: center; margin-bottom: 2rem;">Core Technology & Features</h2>
        <p style="text-align: center; font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 2rem;">
            Our platform is built on a sophisticated, multi-layered technology stack
        </p>
    </div>
    """, unsafe_allow_html=True)

    # AI & Analytical Models
    st.markdown("""
    <div class="feature-card" style="margin-bottom: 2rem;">
        <h3>🧠 AI & Analytical Models</h3>
        <ul style="font-size: 1.1rem; line-height: 1.8;">
            <li><strong>Multi-Agent System (Deepagent):</strong> Utilizes a team of specialized AI agents, orchestrated by <strong>LangGraph</strong>, to perform deep, collaborative research.</li>
            <li><strong>ARIMA Forecasting:</strong> Employs time-series analysis to project potential price movements and inform recommendations.</li>
            <li><strong>Retrieval-Augmented Generation (RAG):</strong> Enriches LLM responses with proprietary data from financial research papers, providing unparalleled context and accuracy.</li>
            <li><strong>Comprehensive Analysis:</strong> Integrates fundamental, quantitative, dividend, and news sentiment analysis into a single, unified view.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Architecture & User Experience
    st.markdown("""
    <div class="feature-card" style="margin-bottom: 3rem;">
        <h3>🔧 Architecture & User Experience</h3>
        <ul style="font-size: 1.1rem; line-height: 1.8;">
            <li><strong>Reactive Agent Design:</strong> Ensures a fluid, real-time, and interactive research workflow.</li>
            <li><strong>Portfolio Analysis:</strong> Empowers users to evaluate up to 10 stocks in parallel.</li>
            <li><strong>PDF Export:</strong> Delivers professional, shareable research reports on demand.</li>
            <li><strong>Cloud-Native:</strong> Deployed on <strong>AWS ECS Fargate</strong> for scalability, reliability, and security.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # How It Works Section (Original Content with Enhanced Styling)
    st.markdown("""
    <div style="background: var(--bg-secondary); padding: 3rem 2rem; border-radius: 16px; margin: 3rem 0;">
        <h2 style="text-align: center; margin-bottom: 2rem;">How It Works</h2>
        <div style="font-size: 1.1rem; line-height: 1.8;">
            <div style="display: flex; align-items: center; margin-bottom: 1.5rem; padding: 1rem; background: var(--card-bg); border-radius: 8px;">
                <div style="font-size: 2rem; margin-right: 1rem;">1️⃣</div>
                <div><strong>Login & Access:</strong> Securely enter the AI Co-Pilot dashboard.</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 1.5rem; padding: 1rem; background: var(--card-bg); border-radius: 8px;">
                <div style="font-size: 2rem; margin-right: 1rem;">2️⃣</div>
                <div><strong>Choose Your Path:</strong> Select the <strong>Deepagent</strong> for forecasting or the <strong>Analysis Suite</strong> for broad research.</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 1.5rem; padding: 1rem; background: var(--card-bg); border-radius: 8px;">
                <div style="font-size: 2rem; margin-right: 1rem;">3️⃣</div>
                <div><strong>Define Scope:</strong> Input stocks, time periods, and toggle RAG for customized analysis.</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 1.5rem; padding: 1rem; background: var(--card-bg); border-radius: 8px;">
                <div style="font-size: 2rem; margin-right: 1rem;">4️⃣</div>
                <div><strong>Receive Insights:</strong> Get comprehensive reports, visualizations, and actionable recommendations.</div>
            </div>
            <div style="display: flex; align-items: center; padding: 1rem; background: var(--card-bg); border-radius: 8px;">
                <div style="font-size: 2rem; margin-right: 1rem;">5️⃣</div>
                <div><strong>Export & Share:</strong> Download your findings as a polished PDF document.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Final CTA
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <h2>Ready to Experience the Future of Stock Research?</h2>
        <p style="font-size: 1.2rem; color: var(--text-secondary); margin-bottom: 2rem;">
            Join investors who are already making smarter decisions with AI-powered insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # col1, col2, col3 = st.columns([1.5, 1, 1.5])
    # with col2:
    #     if st.button("🚀 Start Your AI-Powered Research", type="primary", key="about_final_cta"):
    #         st.session_state.show_login_modal = True
    #         st.rerun()
    # if st.session_state.get("show_login_modal", False):
    #     show_modal_spinner("Loading login page...")
    #     st.session_state.show_login_modal = False
    #     st.session_state.show_login = True
    #     st.rerun()

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("🚀 Start Your AI-Powered Research", type="primary", key="about_final_cta"):
            st.session_state.show_login_modal = True
            st.rerun()
    if st.session_state.get("show_login_modal", False):
        show_modal_spinner("Loading login page...")
        st.session_state.show_login_modal = False
        st.session_state.show_login = True
        st.rerun()


def main():
    """Main function to integrate with your existing Streamlit app"""

    # Initialize session state
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'landing'

    # Always render the navigation sidebar first to keep it at the top
    with st.sidebar:
        # Show Welcome and Logout only if authenticated
        if st.session_state.get("authenticated", False) and st.session_state.get("username"):

            st.markdown(
                f"""
                <div style="
                    background: var(--bg-secondary);
                    border: 2px solid var(--primary-color);
                    border-radius: 18px;
                    padding: 18px 20px;
                    margin-bottom: 18px;
                    box-shadow: 0 4px 12px rgba(33,150,243,0.07);
                    text-align: center;
                    font-size: 1.22rem;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    position: relative;
                    overflow: hidden;
                    color: var(--text-primary);
                ">
                    <span style="font-size:1.5rem; animation: wave 1.2s infinite;">👋</span>
                    Welcome <span style="color:var(--primary-color);">{st.session_state['username']}</span>
                </div>
                <style>
                @keyframes wave {{
                    0% {{ transform: rotate(0deg); }}
                    10% {{ transform: rotate(14deg); }}
                    20% {{ transform: rotate(-8deg); }}
                    30% {{ transform: rotate(14deg); }}
                    40% {{ transform: rotate(-4deg); }}
                    50% {{ transform: rotate(10deg); }}
                    60% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(0deg); }}
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            if st.button("🚪 Logout", key="sidebar_logout_btn", help="Click to logout"):
                from mcp_client.auth.auth_manager import AuthManager
                AuthManager().logout()

        st.markdown("### Navigation")
        if st.button("🏠 Home"):
            st.session_state.current_page = 'landing'
            st.session_state.show_login = False
            st.rerun()
        if st.button("📊 Demo"):
            st.session_state.current_page = 'demo'
            st.session_state.show_login = False
            st.rerun()
        if st.button("🔧 Platform Mechanics"):
            st.session_state.current_page = 'how_it_works'
            st.session_state.show_login = False
            st.rerun()
        if st.button("ℹ️ About"):
            st.session_state.current_page = 'about'
            st.session_state.show_login = False
            st.rerun()

    # URL parameter handling for different pages

    # URL parameter handling for different pages

    query_params = st.query_params
    if 'page' in query_params:
        page = query_params['page'][0]
        if page in ['demo', 'about', 'how_it_works']:
            st.session_state.current_page = page

    # Route to appropriate page
    if st.session_state.show_login:
        # Your existing login/main app logic goes here
        # It will add its own sidebar content below the navigation
        show_main_app()
    else:
        # SEO-friendly entry points
        if st.session_state.current_page == 'demo':
            create_demo_page()
        elif st.session_state.current_page == 'about':
            create_about_page()
        elif st.session_state.current_page == 'how_it_works':
            create_how_it_works_page()
        else:
            create_landing_page()


def show_main_app():
    """Import and run your existing stock agent application"""
    try:

        # Add the project root to Python path if needed
        import sys
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.append(project_root)

        # Initialize ALL session state variables that your app expects
        # This check ensures state is set only once per session.
        if 'session_initialized' not in st.session_state:
            # Core RAG and environment variables
            # RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
            # st.session_state.rag_enabled = RAG_ENABLED

            # Unique ID for execution session
            import uuid
            unique_id = uuid.uuid4().hex[0:8]
            st.session_state["unique_id"] = unique_id

            # Session timeout tracking
            import time
            st.session_state["last_activity"] = int(time.time() * 1000)
            st.session_state["session_expired"] = False

            # Authentication state (since they passed the landing page)
            st.session_state["authentication_status"] = True
            st.session_state["username"] = "authenticated_user"

            # Initialize agent response state
            st.session_state["agent_thinking"] = ""
            st.session_state["last_content"] = ""
            st.session_state["last_content_tab1"] = ""
            st.session_state["last_content_tab2"] = ""
            st.session_state["last_message_time"] = ""
            st.session_state["spinner_active"] = ""
            st.session_state["pending_prompt"] = ""
            st.session_state["pending_action"] = ""
            st.session_state["rag_enabled"] = False
            # st.session_state['session_initialized'] = True
            # st.session_state.current_page= ""
            # st.session_state["tab1_text_input"] = ""
            # st.session_state["tab1_input"] = ""

            # Common chat/analysis state variables
            session_defaults = {
                'messages': [],
                'chat_history': [],
                'current_stock': '',
                'analysis_complete': False,
                'research_results': {},
                'selected_model': 'gpt-4',
                'conversation_id': None,
                'pdf_uploaded': False,
                'portfolio_data': {},
                'last_query': '',
                'agent_state': 'ready',
                # 'rag_toggle': RAG_ENABLED,
            }

            for key, default_value in session_defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

            # Mark session as initialized
            st.session_state['session_initialized'] = True

        # Import your existing stock agent file
        from mcp_client.stock_agent_combined_ui_working_copy import main_app

        # Hide the navigation sidebar for authenticated users to prevent layout conflicts
        st.markdown("""
        <style>
        /* Hide the navigation elements when in authenticated mode */
        .main-navigation { display: none !important; }

        /* Ensure sidebar content flows properly */
        .sidebar-content {
            padding-top: 1rem;
            max-height: 100vh;
            overflow-y: auto;
        }

        /* Prevent sidebar elements from shifting */
        .stSidebar > div:first-child {
            overflow: hidden;
        }

        /* Style the logout button to be less prominent */
        div[data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #f0f2f6;
            border: 1px solid #d3d3d3;
            color: var(--text-secondary);
            font-size: 0.8rem;
            padding: 0.25rem 0.75rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Call your existing main_app function directly
        main_app()

    except ImportError as e:
        st.error(f"Unable to load stock agent application: {e}")
        st.markdown(f"Import error details: {str(e)}")
        st.markdown("Please check that mcp_client/stock_agent_combined_ui_working_copy.py is available.")

        # Debug info
        with st.expander("Debug Information"):
            st.write("Current working directory:", os.getcwd())
            st.write("Python path:", sys.path)
            st.write("Files in mcp_client/:",
                     os.listdir("mcp_client") if os.path.exists("mcp_client") else "Directory not found")

        # Fallback logout option
        if st.button("← Back to Home"):
            st.session_state.show_login = False
            st.session_state.current_page = 'landing'
            st.rerun()

    except AttributeError as e:
        st.error(f"Session state initialization error: {e}")
        st.markdown("Some session state variables are missing. Let's initialize them.")

        # Show current session state for debugging
        with st.expander("Session State Debug"):
            st.write("Current session state keys:", list(st.session_state.keys()))
            st.write("Error details:", str(e))

            # Show which specific variable is missing
            error_str = str(e)
            if "has no attribute" in error_str:
                missing_var = error_str.split("has no attribute '")[1].split("'")[0]
                st.write(f"Missing variable: {missing_var}")

        # Manual session state reset
        if st.button("Reset Session & Retry"):
            # Clear all session state except navigation
            for key in list(st.session_state.keys()):
                if key not in ['show_login', 'current_page']:
                    del st.session_state[key]
            st.rerun()

        # Back to home option
        if st.button("← Back to Home"):
            st.session_state.show_login = False
            st.session_state.current_page = 'landing'
            st.rerun()

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.markdown("An unexpected error occurred while loading the application.")

        with st.expander("Full Error Details"):
            import traceback
            st.code(traceback.format_exc())

        if st.button("← Back to Home"):
            st.session_state.show_login = False
            st.session_state.current_page = 'landing'
            st.rerun()


# Enhanced sitemap.xml content for your separate ECS container
def generate_sitemap_xml():
    """Generate sitemap.xml content for your separate ECS container"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://rajeshranjan.click/</loc>
        <lastmod>2025-08-30</lastmod>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>https://rajeshranjan.click/?page=demo</loc>
        <lastmod>2025-08-30</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.9</priority>
    </url>
    <url>
        <loc>https://rajeshranjan.click/?page=about</loc>
        <lastmod>2025-08-30</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.8</priority>
    </url>
</urlset>"""


def generate_robots_txt():
    """Generate robots.txt content"""
    return """User-agent: *
Allow: /
Allow: /?page=demo
Allow: /?page=about
Disallow: /api/
Disallow: /_stcore/

Sitemap: https://rajeshranjan.click/sitemap.xml
"""


if __name__ == "__main__":
    main()