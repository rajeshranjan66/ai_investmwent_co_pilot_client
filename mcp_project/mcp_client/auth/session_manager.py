# FIXED: session_manager.py - Enhanced session management

import time
from collections import OrderedDict
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()


class SessionManager:
    _instance = None
    _max_sessions = int(os.getenv('MAX_SESSIONS', 2))  # Changed to 2
    #_session_timeout = int(os.getenv('SESSION_TIMEOUT', 10))
    _sessions = OrderedDict()
    _session_timeout_minutes = int(os.getenv('SESSION_TIMEOUT', 10))
    _session_timeout = _session_timeout_minutes * 60  # seconds for cleanup

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def can_create_session(cls):
        """Check if a new session can be created (with cleanup first)"""
        cls.cleanup_old_sessions()
        available_slots = cls._max_sessions - len(cls._sessions)
        print(
            f"DEBUG: can_create_session - Available slots: {available_slots}, Max: {cls._max_sessions}, Current: {len(cls._sessions)}")
        return len(cls._sessions) < cls._max_sessions

    @classmethod
    def create_session(cls, session_id):
        """Create a new session - only call this AFTER successful auth"""
        print(f"DEBUG: create_session called for session_id: {session_id}")

        # Clean up expired sessions first
        cls.cleanup_old_sessions()

        # Double-check capacity (critical for race conditions)
        if len(cls._sessions) >= cls._max_sessions:
            print(f"DEBUG: Session creation failed - at capacity ({len(cls._sessions)}/{cls._max_sessions})")
            return False

        # Create the session
        cls._sessions[session_id] = time.time()
        print(f"DEBUG: Session {session_id} created successfully. Total sessions: {len(cls._sessions)}")
        return True

    @classmethod
    def remove_session(cls, session_id):
        """Remove a specific session"""
        if session_id in cls._sessions:
            cls._sessions.pop(session_id)
            print(f"DEBUG: Session {session_id} removed. Total sessions: {len(cls._sessions)}")
            return True
        return False

    @classmethod
    def get_active_sessions_count(cls):
        """Get count of currently active sessions (after cleanup)"""
        cls.cleanup_old_sessions()
        return len(cls._sessions)

    @classmethod
    def cleanup_old_sessions(cls):
        """Clean up expired sessions"""
        current_time = time.time()
        sessions_to_remove = []

        for session_id, created_time in cls._sessions.items():
            if current_time - created_time > cls._session_timeout:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            cls._sessions.pop(session_id)
            print(f"DEBUG: Expired session {session_id} cleaned up")

        if sessions_to_remove:
            print(f"DEBUG: Cleaned up {len(sessions_to_remove)} expired sessions. Current total: {len(cls._sessions)}")

    @staticmethod
    def get_remaining_time(session_id: str, formatted: bool = True):
        """Get remaining time for a session"""
        if "session_start_times" not in st.session_state:
            st.session_state.session_start_times = {}

        if session_id not in st.session_state.session_start_times:
            st.session_state.session_start_times[session_id] = time.time()

        elapsed = time.time() - st.session_state.session_start_times[session_id]
        total_duration_secs = SessionManager._session_timeout_minutes * 60
        remaining_seconds = total_duration_secs - elapsed

        if remaining_seconds < 0:
            remaining_seconds = 0

        if not formatted:
            return remaining_seconds

        # minutes, seconds = divmod(remaining_seconds, 60)
        # formatted_time = f"{minutes:02}:{seconds:02}"

        remaining_int = int(remaining_seconds)
        minutes = remaining_int // 60
        seconds = remaining_int % 60

        # Zero-pad with two digits
        formatted_time = f"{minutes:02d}:{seconds:02d}"
        return formatted_time, remaining_int

    @classmethod
    def refresh_session(cls, session_id):
        """Refresh the session timestamp on user activity"""
        if session_id in cls._sessions:
            cls._sessions[session_id] = time.time()
            return True
        return False

    @classmethod
    def get_session_info(cls):
        """Get debug information about current sessions"""
        cls.cleanup_old_sessions()
        return {
            "active_sessions": len(cls._sessions),
            "max_sessions": cls._max_sessions,
            "sessions": list(cls._sessions.keys()),
            "can_create": cls.can_create_session()
        }