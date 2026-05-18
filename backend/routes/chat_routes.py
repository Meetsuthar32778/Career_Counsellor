"""
chat_routes.py - Chat API Endpoints
=====================================
Handles starting sessions, sending messages, and retrieving
the adaptive questioning flow.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from typing import Optional
import uuid
from datetime import datetime

from database.database import get_db
from database.models import User, ChatSession, ChatHistory
from database.schemas import ChatMessageRequest, ChatMessageResponse, StartSessionResponse
from utils.security import verify_token
from services.chat_engine import AdaptiveQuestionEngine

# ---------------------------------------------------------------------------
# Router Setup
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/chat", tags=["Chat"])

# ---------------------------------------------------------------------------
# In-memory session engines (maps session_id -> AdaptiveQuestionEngine)
# In production, this would be stored in Redis or similar.
# ---------------------------------------------------------------------------
active_engines = {}


def get_current_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Dependency to extract and validate the current user from JWT token.

    Args:
        authorization: The Authorization header value (Bearer <token>).
        db: Database session dependency.

    Returns:
        The User ORM object.

    Raises:
        HTTPException 401: If token is missing or invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in."
        )

    token = authorization.split(" ")[1]
    payload = verify_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again."
        )

    user = db.query(User).filter(User.username == payload.get("sub")).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found."
        )
    return user


@router.post("/start", response_model=StartSessionResponse)
def start_session(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Start a new career counselling chat session.

    Creates a new ChatSession record and initializes the
    AdaptiveQuestionEngine for this session.

    Returns:
        Session ID, welcome message, and the first question.
    """
    # Create a new session in the database
    session_token = str(uuid.uuid4())
    new_session = ChatSession(
        user_id=user.id,
        session_token=session_token,
        question_count=0,
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    # Initialize the adaptive questioning engine
    engine = AdaptiveQuestionEngine()
    active_engines[new_session.id] = engine

    welcome = engine.get_welcome_message()
    first_question, first_suggestions = engine.get_first_question()

    # Save the bot messages to chat history
    db.add(ChatHistory(session_id=new_session.id, role="bot", message=welcome))
    db.add(ChatHistory(session_id=new_session.id, role="bot", message=first_question))
    db.commit()

    return StartSessionResponse(
        session_id=new_session.id,
        welcome_message=welcome,
        first_question=first_question,
        suggestions=first_suggestions,
    )


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Process a user message and return the chatbot's adaptive response.

    Workflow:
    1. Validate the session belongs to the user.
    2. Pass the answer to the AdaptiveQuestionEngine.
    3. Get the next question or signal completion.
    4. Save both messages to chat history.

    Args:
        request: Contains the user message and session_id.

    Returns:
        The bot's next question, progress info, and detected interests.
    """
    session_id = request.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    # Validate session
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id,
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not session.is_active:
        raise HTTPException(status_code=400, detail="This session has already been completed.")

    # Get or recreate the engine
    engine = active_engines.get(session_id)
    if engine is None:
        # Rebuild engine state from chat history (for server restarts)
        engine = AdaptiveQuestionEngine()
        # Replay previous answers
        history = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id,
            ChatHistory.role == "user",
        ).order_by(ChatHistory.id).all()
        for msg in history:
            engine.process_answer_and_get_next(msg.message)
        active_engines[session_id] = engine

    # Save user message
    db.add(ChatHistory(session_id=session_id, role="user", message=request.message))

    # Process answer and get next question in a background threadpool to unblock async event loop
    def ml_task():
        return engine.process_answer_and_get_next(request.message)

    next_question, next_suggestions, is_complete = await run_in_threadpool(ml_task)

    # Update session question count
    session.question_count = engine.question_number - 1
    current_q, total_q = engine.get_progress()

    if is_complete:
        # Session is complete - mark it
        session.is_active = 0
        session.ended_at = datetime.utcnow()
        bot_message = (
            "Thank you for sharing your thoughts so openly! 🎉\n\n"
            "I've gathered enough information to analyze your interests and strengths. "
            "Let me prepare your personalized career recommendation..."
        )
        suggestions_to_send = None
    else:
        bot_message = next_question
        suggestions_to_send = next_suggestions

    # Save bot message
    db.add(ChatHistory(session_id=session_id, role="bot", message=bot_message))
    db.commit()

    return ChatMessageResponse(
        bot_message=bot_message,
        session_id=session_id,
        question_number=current_q,
        total_questions=total_q,
        is_complete=is_complete,
        detected_interests=engine.get_detected_interests(),
        suggestions=suggestions_to_send,
        low_confidence_warning=engine.low_confidence_warning,
    )


@router.get("/history/{session_id}")
def get_chat_history(
    session_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve the full chat history for a given session.

    Args:
        session_id: The ID of the chat session.

    Returns:
        List of messages with role, text, and timestamp.
    """
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id,
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.id).all()

    return {
        "session_id": session_id,
        "messages": [
            {
                "role": msg.role,
                "message": msg.message,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            for msg in messages
        ],
        "is_active": bool(session.is_active),
        "started_at": session.started_at.isoformat() if session.started_at else None,
    }


@router.get("/sessions")
def get_user_sessions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all chat sessions for the current user.

    Returns:
        List of session summaries with ID, status, and timestamps.
    """
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user.id
    ).order_by(ChatSession.started_at.desc()).all()

    return {
        "sessions": [
            {
                "id": s.id,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "is_active": bool(s.is_active),
                "question_count": s.question_count,
            }
            for s in sessions
        ]
    }
