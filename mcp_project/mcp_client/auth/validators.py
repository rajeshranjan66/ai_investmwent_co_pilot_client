# auth/validators.py
import re
from typing import Tuple


class EmailValidator:
    @staticmethod
    def validate(email: str) -> Tuple[bool, str]:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not email:
            return False, "Email is required"
        if not re.match(pattern, email):
            return False, "Invalid email format"
        return True, "Valid email"


class PasswordValidator:
    @staticmethod
    def validate(password: str) -> Tuple[bool, str]:
        if not password:
            return False, "Password is required"
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number"
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character"
        return True, "Strong password"


class UsernameValidator:
    @staticmethod
    def validate(username: str, auth_manager) -> Tuple[bool, str]:
        if not username:
            return False, "Username is required"
        if len(username) < 4:
            return False, "Username must be at least 4 characters long"
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"

        # Check username availability
        users = auth_manager.load_users()
        if username in users["usernames"]:
            return False, "Username already taken"
        return True, "Username available"


class NameValidator:
    @staticmethod
    def validate(name: str) -> Tuple[bool, str]:
        if not name:
            return False, "Full name is required"
        if len(name.strip()) < 2:
            return False, "Name is too short"
        if not re.match(r'^[a-zA-Z\s-]+$', name):
            return False, "Name can only contain letters, spaces, and hyphens"
        return True, "Valid name"


class ConfirmPasswordValidator:
    @staticmethod
    def validate(password: str, confirm_password: str) -> Tuple[bool, str]:
        if not confirm_password:
            return False, "Please confirm your password"
        if password != confirm_password:
            return False, "🔐 Your passwords are not twins yet. Make them match for a perfect pair!"
        return True, "Passwords match"
