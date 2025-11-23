# auth_manager.py
import yaml
import streamlit_authenticator as stauth
import bcrypt
from typing import Dict, Optional, Tuple
import logging
from .config import USER_FILE, COOKIE_NAME, COOKIE_KEY, COOKIE_EXPIRY_DAYS
import streamlit as st

logger = logging.getLogger(__name__)


class AuthManager:
    def __init__(self):
        self._ensure_user_file_exists()

    def _ensure_user_file_exists(self):
        if not USER_FILE.exists():
            initial_data = {
                "usernames": {},
                "names": {},
                "passwords": {},
                "emails": {}
            }
            with open(USER_FILE, "w") as f:
                yaml.dump(initial_data, f)

    def load_users(self) -> dict:
        try:
            with open(USER_FILE) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return {
                "usernames": {},
                "names": {},
                "passwords": {},
                "emails": {}
            }

    def save_users(self, users: dict):
        try:
            with open(USER_FILE, "w") as f:
                yaml.dump(users, f)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
            raise

    def register_user(self, name: str, username: str, password: str, email: str) -> Tuple[bool, str]:
        try:
            users = self.load_users()

            if username in users["usernames"]:
                return False, "Username already exists"

            # Generate password hash using bcrypt
            password_bytes = password.encode('utf-8')
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password_bytes, salt)

            # Store the hash as a string
            users["usernames"][username] = username
            users["names"][username] = name
            users["passwords"][username] = hashed_password.decode('utf-8')  # Convert bytes to string
            users["emails"][username] = email

            self.save_users(users)
            return True, "Registration successful"

        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return False, f"Registration failed: {str(e)}"

    def verify_password(self, stored_password: str, provided_password: str) -> bool:
        try:
            stored_bytes = stored_password.encode('utf-8')
            provided_bytes = provided_password.encode('utf-8')
            return bcrypt.checkpw(provided_bytes, stored_bytes)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def get_authenticator(self) -> stauth.Authenticate:
        try:
            users = self.load_users()
            return stauth.Authenticate(
                names=users["names"],
                usernames=users["usernames"],
                passwords=users["passwords"],
                cookie_name=COOKIE_NAME,
                key=COOKIE_KEY,
                cookie_expiry_days=COOKIE_EXPIRY_DAYS
            )
        except Exception as e:
            logger.error(f"Error creating authenticator: {e}")
            raise

    # auth_manager.py

    def logout(self) -> None:
        """Handle user logout with complete session state cleanup"""
        try:
            # Authentication state
            st.session_state.authenticated = False
            st.session_state.authentication_status = False

            # Clear user information
            if 'username' in st.session_state:
                del st.session_state['username']

            # Explicitly clear agent/session keys
            explicit_keys = [
                "agent_thinking", "last_content", "last_message_time",
                "stock_rows", "stocks_data", "cash_input", "authentication_status",
                # Chat and Analysis State
                "messages", "chat_history", "current_stock", "analysis_complete",
                "research_results", "selected_model", "conversation_id",
                "pdf_uploaded", "portfolio_data", "last_query", "agent_state",
                # Agent Response State
                "last_content_tab1", "last_content_tab2", "spinner_active",
                "pending_prompt", "pending_action", "rag_enabled",
                # Session Management
                "session_id", "session_initialized", "unique_id",
                "last_activity", "session_expired", "session_start_times",
                # UI State
                "show_login", "show_login_modal", "current_page"
            ]

            for key in explicit_keys:
                if key in st.session_state:
                    del st.session_state[key]

            # Clear all session states that start with specific prefixes
            prefix_patterns = (
                'chat_',
                'captcha_',
                'login_',
                'register_',
                'app_',
                'form_',
                'input_',
                'last_',  # Added for last_content related keys
                'stock_',  # Added for stock related keys
                'agent_'  # Added for agent related keys
            )

            keys_to_clear = [
                key for key in st.session_state.keys()
                if key.startswith(prefix_patterns)
            ]
            for key in keys_to_clear:
                del st.session_state[key]

            # Reset UI state
            st.session_state.current_page = 'landing'

            logger.info("User logged out successfully")
            st.rerun()
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            raise

    # def logout(self) -> None:
    #     """Handle user logout with complete session state cleanup"""
    #     try:
    #         st.session_state.authenticated = False
    #
    #         # Clear user information
    #         if 'username' in st.session_state:
    #             del st.session_state['username']
    #
    #         # Explicitly clear agent/session keys
    #         for key in [
    #             "agent_thinking", "last_content", "last_message_time",
    #             "stock_rows", "stocks_data", "cash_input", "authentication_status"
    #         ]:
    #             if key in st.session_state:
    #                 del st.session_state[key]
    #
    #         # Clear all session states that start with specific prefixes
    #         keys_to_clear = [
    #             key for key in st.session_state.keys()
    #             if key.startswith((
    #                 'chat_',
    #                 'captcha_',
    #                 'login_',
    #                 'register_',
    #                 'app_',
    #                 'form_',
    #                 'input_'
    #             ))
    #         ]
    #         for key in keys_to_clear:
    #             del st.session_state[key]
    #
    #         logger.info("User logged out successfully")
    #         st.rerun()
    #     except Exception as e:
    #         logger.error(f"Error during logout: {e}")
    #         raise


# mcp_client/auth/auth_manager.py
import streamlit as st
import time

def check_session_timeout(timeout_seconds=15):
    now = time.time()
    if "last_activity" in st.session_state:
        elapsed = now - st.session_state["last_activity"]
        if elapsed > timeout_seconds:
            # Clear session and redirect to login
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["session_expired"] = True
            st.rerun()
    st.session_state["last_activity"] = now