# mcp_client/ui/ui_spinner.py
import streamlit as st
import time

def inject_spinner_css():
    st.markdown("""
    <style>
    #modal-spinner {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(30, 30, 30, 0.7); z-index: 9999; display: flex;
        align-items: center; justify-content: center;
    }
    .loader {
        border: 8px solid #f3f3f3; border-top: 8px solid #3498db;
        border-radius: 50%; width: 60px; height: 60px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg);}
        100% { transform: rotate(360deg);}
    }
    </style>
    """, unsafe_allow_html=True)

def show_modal_spinner(message="Loading..."):
    inject_spinner_css()
    st.markdown(f"""
    <div id="modal-spinner">
        <div style="text-align: center;">
            <div class="loader"></div>
            <p style="color: white; font-size: 1.3rem; margin-top: 1rem;">{message}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(.6)