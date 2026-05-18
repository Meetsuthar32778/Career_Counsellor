"""
security.py - Authentication & Security Utilities
===================================================
Handles password hashing, JWT token creation, and
token verification for user authentication.
"""

import hashlib
import secrets
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
SECRET_KEY = "career_counsellor_secret_key_2024_ai_lab_project"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120  # Token valid for 2 hours



def hash_password(password: str) -> str:
    """
    Hash a plain-text password using SHA-256 with a random salt.
    
    Args:
        password: The plain-text password from user input.
    
    Returns:
        The salted hash string in format 'salt$hash'.
    """
    salt = secrets.token_hex(16)
    pw_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${pw_hash}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against its hashed version.
    
    Args:
        plain_password: The password entered by the user.
        hashed_password: The stored hashed password from the database.
    
    Returns:
        True if the password matches, False otherwise.
    """
    try:
        salt, stored_hash = hashed_password.split("$", 1)
        pw_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
        return pw_hash == stored_hash
    except ValueError:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token with the given payload.
    
    Args:
        data: Dictionary containing claims (e.g., {"sub": username}).
        expires_delta: Optional custom expiration time.
    
    Returns:
        Encoded JWT token string.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT token.
    
    Args:
        token: The JWT token string to verify.
    
    Returns:
        The decoded payload dictionary, or None if invalid/expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
