import streamlit as st

def inject_animations_css():
    st.markdown("""
    <style>
    /* Fade-in for tab content */
    .stTabs [data-baseweb="tab-panel"] {
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInTab 0.5s forwards;
    }
    @keyframes fadeInTab {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    /* Slide-in for error/success messages */
    .stAlert {
        opacity: 0;
        transform: translateX(-30px);
        animation: slideInMsg 0.4s forwards;
    }
    @keyframes slideInMsg {
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    </style>
    """, unsafe_allow_html=True)