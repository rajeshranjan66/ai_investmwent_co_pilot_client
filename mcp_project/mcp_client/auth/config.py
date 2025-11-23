import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
CONFIG_DIR = Path(__file__).parent.parent / "config"
USER_FILE = CONFIG_DIR / "users.yaml"

# Ensure config directory exists
CONFIG_DIR.mkdir(exist_ok=True)

# Authentication configuration
COOKIE_NAME = "mcp_auth_cookie"
COOKIE_KEY = os.getenv('COOKIE_KEY', 'your-default-secret-key')  # Make sure to set in .env
COOKIE_EXPIRY_DAYS = 30
