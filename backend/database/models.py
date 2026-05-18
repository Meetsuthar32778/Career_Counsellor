"""
models.py - SQLAlchemy ORM Models
==================================
Defines the database tables for users, sessions,
chat history, and prediction results.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.database import Base


class User(Base):
    """
    Stores registered user information.
    Passwords are hashed before storage.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    results = relationship("Result", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    """
    Represents a single counselling conversation session.
    Each session tracks the questions asked and answers given.
    """
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(100), unique=True, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = completed
    question_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")
    result = relationship("Result", back_populates="session", uselist=False)


class ChatHistory(Base):
    """
    Stores individual messages (questions and answers) within a session.
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(10), nullable=False)  # "bot" or "user"
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class Result(Base):
    """
    Stores the final career prediction result for a session.
    Includes the predicted field, confidence score, and explanation.
    """
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, unique=True)
    predicted_field = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    all_scores = Column(Text, nullable=True)   # JSON string of all field scores
    explanation = Column(Text, nullable=True)   # Detailed explanation text
    strengths = Column(Text, nullable=True)     # JSON string of detected strengths
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="results")
    session = relationship("ChatSession", back_populates="result")
