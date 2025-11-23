import uuid

import streamlit as st
from functools import wraps
from typing import Callable, Any
import sys
import os
import uuid
import streamlit as st







def require_auth(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if not st.session_state.get('authenticated', False):
            from .auth_ui import AuthUI
            auth_ui = AuthUI()
            # Pass the callback with correct signature
            auth_ui.render_login_register(
                on_auth=lambda username: setattr(st.session_state, 'username', username)
            )
            return None
        return func(*args, **kwargs)
    return wrapper




