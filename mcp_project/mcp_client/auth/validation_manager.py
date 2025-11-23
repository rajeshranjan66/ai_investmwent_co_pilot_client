# auth/validation_manager.py
from typing import Dict, Any
from .validators import (EmailValidator, PasswordValidator, UsernameValidator,
                       NameValidator, ConfirmPasswordValidator)

class ValidationManager:
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.email_validator = EmailValidator()
        self.password_validator = PasswordValidator()
        self.username_validator = UsernameValidator()
        self.name_validator = NameValidator()
        self.confirm_password_validator = ConfirmPasswordValidator()

    def validate_registration_field(self, field_name: str, value: str, password: str = None) -> Dict[str, Any]:
        if field_name == "email":
            is_valid, message = self.email_validator.validate(value)
        elif field_name == "password":
            is_valid, message = self.password_validator.validate(value)
        elif field_name == "confirm_password":
            is_valid, message = self.confirm_password_validator.validate(password, value)
        elif field_name == "username":
            is_valid, message = self.username_validator.validate(value, self.auth_manager)
        elif field_name == "name":
            is_valid, message = self.name_validator.validate(value)
        else:
            return {"valid": False, "message": "Unknown field"}

        return {
            "valid": is_valid,
            "message": message
        }
