from typing import Dict
import re


class FormValidator:
    @staticmethod
    def validate_username(username: str) -> Dict[str, bool | str]:
        """Validate username field"""
        if not username:
            return {"valid": False, "error": "Username is required"}
        if len(username) < 4:
            return {"valid": False, "error": "Username must be at least 4 characters long"}
        if not username.isalnum():
            return {"valid": False, "error": "Username should contain only letters and numbers"}
        return {"valid": True, "error": ""}

    @staticmethod
    def validate_name(name: str) -> Dict[str, bool | str]:
        """Validate name field"""
        if not name:
            return {"valid": False, "error": "Name is required"}
        if not all(char.isalpha() or char.isspace() for char in name):
            return {"valid": False, "error": "Name should contain only letters and spaces"}
        return {"valid": True, "error": ""}

    @staticmethod
    def validate_email(email: str) -> Dict[str, bool | str]:
        """Validate email field"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not email:
            return {"valid": False, "error": "Email is required"}
        if not re.match(email_pattern, email):
            return {"valid": False, "error": "Invalid email format"}
        return {"valid": True, "error": ""}

    @staticmethod
    def validate_password(password: str) -> Dict[str, bool | str]:
        """Validate password field"""
        if not password:
            return {"valid": False, "error": "Password is required"}
        if len(password) < 8:
            return {"valid": False, "error": "Password must be at least 8 characters long"}
        if not any(char.isupper() for char in password):
            return {"valid": False, "error": "Password must contain at least one uppercase letter"}
        if not any(char.islower() for char in password):
            return {"valid": False, "error": "Password must contain at least one lowercase letter"}
        if not any(char.isdigit() for char in password):
            return {"valid": False, "error": "Password must contain at least one number"}
        return {"valid": True, "error": ""}
