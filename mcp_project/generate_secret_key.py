# Generate a secure key
import secrets
secure_key = secrets.token_hex(16)  # Generates a 32-character hexadecimal string
print(secure_key)  # Copy this value to your .env file
