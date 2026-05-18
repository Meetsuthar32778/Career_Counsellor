"""
auth_routes.py - Authentication API Endpoints
===============================================
Handles user registration, login, and token-based authentication.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.database import get_db
from database.models import User
from database.schemas import (
    UserRegister,
    UserLogin,
    ForgotPasswordRequest,
    TokenResponse,
    UserResponse,
)
from utils.security import hash_password, verify_password, create_access_token

# ---------------------------------------------------------------------------
# Router Setup
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user account.

    Steps:
    1. Check if username or email already exists.
    2. Hash the password.
    3. Create the user record in the database.

    Args:
        user_data: Registration payload (username, email, password, full_name).
        db: Database session dependency.

    Returns:
        The newly created user data (without password).

    Raises:
        HTTPException 400: If username or email is already taken.
    """
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken. Please choose a different one."
        )

    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered. Please use a different email."
        )

    # Create the new user with hashed password
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@router.post("/login", response_model=TokenResponse)
def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate a user and return a JWT token.

    Steps:
    1. Find the user by username.
    2. Verify the password.
    3. Generate and return a JWT access token.

    Args:
        login_data: Login payload (username, password).
        db: Database session dependency.

    Returns:
        JWT access token and user info.

    Raises:
        HTTPException 401: If credentials are invalid.
    """
    # Find user by username
    user = db.query(User).filter(User.username == login_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

    # Generate JWT token
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        username=user.username,
        user_id=user.id,
    )


@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Reset password using username and registered email.

    Args:
        payload: Forgot password request payload.
        db: Database session dependency.

    Returns:
        Success message after password update.

    Raises:
        HTTPException 404: If matching user is not found.
    """
    user = db.query(User).filter(
        User.username == payload.username,
        User.email == payload.email
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with the provided username and email."
        )

    user.hashed_password = hash_password(payload.new_password)
    db.commit()

    return {"detail": "Password reset successful. Please log in with your new password."}
