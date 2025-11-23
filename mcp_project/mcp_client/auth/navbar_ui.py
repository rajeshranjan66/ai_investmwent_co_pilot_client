# navbar_ui.py
import streamlit as st
from .auth_manager import AuthManager

class NavbarUI:
    def __init__(self):
        self._init_styles()
        # Initialize AuthManager
        self.auth_manager = AuthManager()

    def _init_styles(self):
        self.style = """
            <style>
            .navbar {
                padding: 10px 0;
                margin-bottom: 20px;
            }
            .navbar-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .logout-button {
                background-color: #ff4b4b;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                border: none;
                cursor: pointer;
                font-size: 14px;
            }
            .logout-button:hover {
                background-color: #ff3333;
            }
            .username-display {
                color: #2e6bb7;
                font-size: 14px;
                margin-right: 15px;
            }
            </style>
        """





    def render_navbar(self) -> None:
        st.markdown("""
            <style>
            .navbar {
                padding: 10px 0;
                margin-bottom: 20px;
            }
            .navbar-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .logout-button {
                background-color: #2E86C1;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                border: none;
                cursor: pointer;
                font-size: 14px;
                font-family: inherit;
                margin-right: 8px;
            }
            .logout-button:hover {
                background-color: #21618C;
            }
            .username-display {
                background-color: #2E86C1;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                border: none;
                font-size: 14px;
                font-family: inherit;
                margin-right: 8px;
                display: inline-block;
            }
            </style>
        """, unsafe_allow_html=True)

        # with st.container():
        #     cols = st.columns([2, 6, 1])
        #     with cols[0]:
        #         if 'username' in st.session_state:
        #             st.markdown(
        #                 f"""<div class="username-display">Welcome {st.session_state['username']} !</div>""",
        #                 unsafe_allow_html=True
        #             )
        #     # Middle column left empty for spacing
        #     with cols[2]:
        #         if st.button("🚪 Logout", key="navbar_logout_btn", help="Click to logout"):
        #             self.auth_manager.logout()
