# ui/responsive.py
import streamlit as st

def inject_responsive_css():
    st.markdown("""
    <style>
    @media (max-width: 600px) {
        .main-title h1, .main-title h2 {
            font-size: 22px !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px !important;
            padding: 10px 10px !important;
        }
        div[data-testid="stForm"] {
            max-width: 95vw !important;
            padding: 10px !important;
        }
        div.stButton > button {
            width: 90% !important;
            font-size: 16px !important;
        }
        .navbar-content {
            flex-direction: column !important;
            align-items: flex-start !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)