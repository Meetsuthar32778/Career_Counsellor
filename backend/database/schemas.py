"""
schemas.py - Pydantic Models for Request/Response Validation
=============================================================
Defines data shapes used across the API endpoints.
"""

from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime


# ---------------------------------------------------------------------------
# Authentication Schemas
# ---------------------------------------------------------------------------

class UserRegister(BaseModel):
    """Schema for user registration request."""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login request."""
    username: str
    password: str


class ForgotPasswordRequest(BaseModel):
    """Schema for forgot password request."""
    username: str
    email: str
    new_password: str


class UserResponse(BaseModel):
    """Schema for user data in responses."""
    id: int
    username: str
    email: str
    full_name: Optional[str] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Schema for JWT token response after login."""
    access_token: str
    token_type: str = "bearer"
    username: str
    user_id: int


# ---------------------------------------------------------------------------
# Chat Schemas
# ---------------------------------------------------------------------------

class ChatMessageRequest(BaseModel):
    """Schema for sending a user message to the chatbot."""
    message: str
    session_id: Optional[int] = None


class ChatMessageResponse(BaseModel):
    """Schema for the chatbot's reply."""
    bot_message: str
    session_id: int
    question_number: int
    total_questions: int
    is_complete: bool = False
    detected_interests: Optional[Dict[str, float]] = None
    suggestions: Optional[List[str]] = None
    low_confidence_warning: bool = False


class StartSessionResponse(BaseModel):
    """Schema for starting a new chat session."""
    session_id: int
    welcome_message: str
    first_question: str
    suggestions: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Prediction / Result Schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Schema for the final career prediction result."""
    predicted_field: str
    confidence_score: float
    all_scores: Dict[str, float]
    explanation: str
    strengths: List[str]
    personality_traits: List[str]
    suggested_skills: List[str]
    low_confidence_warning: bool = False


class SessionHistoryResponse(BaseModel):
    """Schema for returning a session's chat history."""
    session_id: int
    messages: List[dict]
    result: Optional[dict] = None
    started_at: Optional[str] = None
