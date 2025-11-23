# auth_ui.py

import streamlit as st
from typing import Optional, Callable
from .auth_manager import AuthManager
from .responsive import inject_responsive_css
inject_responsive_css()
from .animations import inject_animations_css
inject_animations_css()
from .toast import show_toast
from .captcha_utils import render_captcha, reset_captcha
from .form_validators import FormValidator
import uuid

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp_client.auth.session_manager import SessionManager


class AuthUI:
    def __init__(self):
        self.auth_manager = AuthManager()

    def render_login_register(self, on_auth: Callable[[str], None]) -> Optional[str]:
        with st.container():
            # Custom CSS for centering and styling with dark mode support
            st.markdown("""
            <style>
            /* CSS Variables for Light/Dark Mode */
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --text-primary: #000000;
                --text-secondary: #666666;
                --card-bg: #ffffff;
                --card-border: rgba(0, 0, 0, 0.1);
                --card-shadow: 0 2px 4px rgba(0,0,0,0.1);
                --primary-color: #2e6bb7;
                --primary-hover: #235292;
                --input-bg: #ffffff;
                --input-border: #ccc;
                --input-text: #000000;
            }

            @media (prefers-color-scheme: dark) {
                :root {
                    --bg-primary: #0f1116;
                    --bg-secondary: #1e1e1e;
                    --text-primary: #ffffff;
                    --text-secondary: #cccccc;
                    --card-bg: #1e1e1e;
                    --card-border: rgba(255, 255, 255, 0.1);
                    --card-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    --primary-color: #90caf9;
                    --primary-hover: #64b5f6;
                    --input-bg: #2d2d2d;
                    --input-border: #555;
                    --input-text: #ffffff;
                }
            }

            .stHeader {
                position: relative;
            }
            # .stHeader .stToolbar,
            # .stHeader [data-testid="stToolbar"],
            # [data-testid="stToolbar"] {
            #     display: none !important;
            # }

            .main-title {
                text-align: center;
                padding: 20px;
            }
            .main-title h1 {
                color: var(--primary-color);
                font-size: 32px;
                margin-bottom: 0;
            }
            .subtitle {
                color: var(--text-secondary);
                font-size: 18px;
                margin-top: 5px;
            }

            /* Updated Tab Styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 30px;
                justify-content: center;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 20px;
                font-weight: 500;
                padding: 15px 30px;
                color: var(--text-primary) !important;
            }

            div[data-testid="stForm"] {
                max-width: 500px;
                margin: 0 auto;
                padding: 20px;
                background-color: var(--card-bg);
                border-radius: 10px;
                box-shadow: var(--card-shadow);
                border: 1px solid var(--card-border);
            }

            /* Improved button container styling - Applies to ALL form buttons */
            div[data-testid="stFormSubmitButton"] {
                display: flex !important;
                justify-content: center !important;
                width: 100% !important;
                flex-shrink: 0 !important;
            }

            /* Improved button styling - Applies to ALL form buttons */
            div[data-testid="stFormSubmitButton"] > button {
                width: 100% !important;
                min-width: 200px !important;
                max-width: 300px !important;
                box-sizing: border-box !important;
                margin: 10px auto !important;
                background-color: var(--primary-color) !important;
                color: white !important;
                border-radius: 5px !important;
                padding: 10px 20px !important;
                font-weight: 500 !important;
                border: none !important;
                transition: background-color 0.3s !important;
            }
            div[data-testid="stFormSubmitButton"] > button:hover {
                background-color: var(--primary-hover) !important;
                color: white !important;
            }

            /* Style Streamlit input fields for dark mode */
            .stTextInput > div > div > input,
            .stTextInput > div > div > input:focus {
                background-color: var(--input-bg) !important;
                color: var(--input-text) !important;
                border-color: var(--input-border) !important;
            }

            .stTextInput > div > div {
                background-color: var(--input-bg) !important;
                border-color: var(--input-border) !important;
            }

            /* Style labels and text */
            .stTextInput label,
            .stTextInput p {
                color: var(--text-primary) !important;
            }

            /* Style the demo credentials box */
            .demo-credentials {
                text-align: center;
                padding: 20px;
                margin: 20px auto;
                max-width: 600px;
                color: var(--text-secondary);
                font-size: 14px;
                background-color: var(--bg-secondary);
                border-radius: 8px;
                border: 1px solid var(--card-border);
            }

            .demo-credentials span {
                color: var(--primary-color);
                font-weight: 500;
            }
            </style>

            <script>
            // Detect and apply dark mode preference
            function applyDarkMode() {
                if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                    document.documentElement.setAttribute('data-theme', 'dark');
                } else {
                    document.documentElement.setAttribute('data-theme', 'light');
                }
            }

            // Apply on load
            applyDarkMode();

            // Listen for theme changes
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyDarkMode);
            </script>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: var(--primary-color);">🚀AI Investment Co-Pilot</h2>
            <p style="color: var(--text-secondary); font-size: 1.1rem;">
                Multi-Agent Stock Research & Forecasting Suite
            </p>
        </div>
        """, unsafe_allow_html=True)

        if 'authentication_status' not in st.session_state:
            st.session_state['authentication_status'] = None

        # Create centered tabs
        tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Create Account"])

        # Add this at the bottom of your render_login_register method
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.markdown("""
            <div class="demo-credentials">
                <strong>Demo Access:</strong> Use <span>stockuser</span> / <span>stockpwd</span> to explore immediately
            </div>
            """, unsafe_allow_html=True)

            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")

                    validation_placeholder = st.empty()
                    captcha_verified_login = render_captcha("login")
                    submitted = st.form_submit_button("Sign In")

                    if submitted:
                        validation_errors = []

                        # Basic validation
                        if not username:
                            validation_errors.append("Username is required")
                        if not password:
                            validation_errors.append("Password is required")

                        if validation_errors:
                            for error in validation_errors:
                                validation_placeholder.error(error)
                            reset_captcha("login")
                        elif not captcha_verified_login:
                            validation_placeholder.error("Please complete the CAPTCHA correctly")
                            reset_captcha("login")
                        else:
                            # CRITICAL FIX: Check session capacity BEFORE authentication
                            if not SessionManager.can_create_session():
                                validation_placeholder.error("⚠️ Session limit reached. Please try again later.")
                                reset_captcha("login")
                                return None

                            # Proceed with authentication
                            users = self.auth_manager.load_users()
                            if username in users["usernames"]:
                                stored_password = users["passwords"][username]
                                if self.auth_manager.verify_password(stored_password, password):

                                    # CRITICAL FIX: Create session IMMEDIATELY after auth success
                                    session_id = str(uuid.uuid4())

                                    # CRITICAL FIX: Verify session creation succeeded
                                    if SessionManager.create_session(session_id):
                                        # Only set session state if session creation succeeded
                                        st.session_state.session_id = session_id
                                        st.session_state.authenticated = True
                                        st.session_state.username = username
                                        st.session_state.authentication_status = True

                                        show_toast("✅ Login successful!", type="success", duration=3000)
                                        on_auth(username)
                                        st.rerun()
                                    else:
                                        validation_placeholder.error("❌ Unable to create session. Please try again.")
                                        reset_captcha("login")
                                else:
                                    validation_placeholder.error("❌ Invalid Username/password")
                                    reset_captcha("login")
                            else:
                                validation_placeholder.error("❌ Invalid Username/password")
                                reset_captcha("login")

            # Register Tab
            with tab2:
                with st.form("register_form"):
                    name = st.text_input("Your Name", placeholder="Enter your full name", key="register_name")
                    email = st.text_input("Email", placeholder="Enter your email", key="register_email")
                    username = st.text_input("Username", placeholder="Choose a username", key="register_username")
                    password = st.text_input("Password", type="password", placeholder="Choose a password",
                                             key="register_password")
                    confirm_password = st.text_input("Re-enter Password", type="password",
                                                     placeholder="Confirm your password",
                                                     key="register_confirm_password")
                    validation_placeholder = st.empty()
                    captcha_verified_register = render_captcha("register")
                    submitted = st.form_submit_button("Sign Up")

                    if submitted:
                        validation_errors = []
                        required_fields = {
                            "Your Name": name,
                            "Email": email,
                            "Username": username,
                            "Password": password,
                            "Re-enter Password": confirm_password
                        }
                        empty_fields = [field for field, value in required_fields.items() if not value]
                        if empty_fields:
                            validation_placeholder.error(
                                f"Please fill the following required fields: {', '.join(empty_fields)}")
                            reset_captcha("register")
                        else:
                            name_validation = FormValidator.validate_name(name)
                            email_validation = FormValidator.validate_email(email)
                            username_validation = FormValidator.validate_username(username)
                            password_validation = FormValidator.validate_password(password)

                            if not name_validation["valid"]:
                                validation_errors.append(name_validation["error"])
                            if not email_validation["valid"]:
                                validation_errors.append(email_validation["error"])
                            if not username_validation["valid"]:
                                validation_errors.append(username_validation["error"])
                            if not password_validation["valid"]:
                                validation_errors.append(password_validation["error"])
                            if password != confirm_password:
                                validation_errors.append("Password is not matching")

                            if validation_errors:
                                for error in validation_errors:
                                    validation_placeholder.error(error)
                                reset_captcha("register")
                            elif not captcha_verified_register:
                                validation_placeholder.error("Please complete the CAPTCHA correctly")
                                reset_captcha("register")
                            else:
                                success, message = self.auth_manager.register_user(name, username, password, email)

                                if success:
                                    show_toast("Sing-up is  successful! Please login with your credentials.",
                                               type="success", duration=4000, position="bottom")  # Longer duration
                                else:
                                    show_toast(message, type="error", duration=3000)  # Error stays at top
                                    reset_captcha("register")

            # Add some spacing at the bottom
            st.markdown("<br><br>", unsafe_allow_html=True)