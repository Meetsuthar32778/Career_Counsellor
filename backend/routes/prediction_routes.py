"""
prediction_routes.py - Career Prediction API Endpoints
========================================================
Handles generating final career predictions and explanation reports.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import Optional
import json

from backend.database.database import get_db
from backend.database.models import User, ChatSession, ChatHistory, Result
from backend.database.schemas import PredictionResponse
from backend.utils.security import verify_token
from backend.model.predictor import get_predictor
from backend.routes.chat_routes import get_current_user, active_engines
from backend.services.chat_engine import AdaptiveQuestionEngine

# ---------------------------------------------------------------------------
# Router Setup
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/predict", tags=["Prediction"])


@router.get("/result/{session_id}", response_model=PredictionResponse)
def get_prediction(
    session_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate and return the career prediction for a completed session.

    Workflow:
    1. Validate the session exists and is completed.
    2. Check if a result already exists (return cached result).
    3. Collect all user answers from chat history.
    4. Pass answers through the CareerPredictor ML pipeline.
    5. Save and return the prediction with explanation.

    Args:
        session_id: The ID of the completed chat session.

    Returns:
        Full prediction result with confidence scores, explanation,
        strengths, and suggested skills.

    Raises:
        HTTPException 400: If the session is still active.
        HTTPException 404: If the session is not found.
    """
    # Validate session
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id,
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.is_active:
        raise HTTPException(
            status_code=400,
            detail="Session is still active. Please complete all questions first."
        )

    # Check if result already exists (avoid re-computing)
    existing_result = db.query(Result).filter(Result.session_id == session_id).first()
    if existing_result:
        return PredictionResponse(
            predicted_field=existing_result.predicted_field,
            confidence_score=existing_result.confidence_score,
            all_scores=json.loads(existing_result.all_scores) if existing_result.all_scores else {},
            explanation=existing_result.explanation or "",
            strengths=json.loads(existing_result.strengths) if existing_result.strengths else [],
            personality_traits=_get_traits(existing_result.predicted_field),
            suggested_skills=_get_skills(existing_result.predicted_field),
        )

    # Collect all user answers from the session
    user_messages = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id,
        ChatHistory.role == "user",
    ).order_by(ChatHistory.id).all()

    answers = [msg.message for msg in user_messages]

    if not answers:
        raise HTTPException(
            status_code=400,
            detail="No answers found in this session."
        )

    # Run the ML prediction pipeline
    predictor = get_predictor()
    result = predictor.predict(answers)

    # Save result to database
    new_result = Result(
        user_id=user.id,
        session_id=session_id,
        predicted_field=result["predicted_field"],
        confidence_score=result["confidence_score"],
        all_scores=json.dumps(result["all_scores"]),
        explanation=result["explanation"],
        strengths=json.dumps(result["strengths"]),
    )
    db.add(new_result)
    db.commit()

    return PredictionResponse(
        predicted_field=result["predicted_field"],
        confidence_score=result["confidence_score"],
        all_scores=result["all_scores"],
        explanation=result["explanation"],
        strengths=result["strengths"],
        personality_traits=result["personality_traits"],
        suggested_skills=result["suggested_skills"],
    )


@router.get("/history")
def get_past_results(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve all past prediction results for the current user.

    Returns:
        List of past results with prediction details.
    """
    results = db.query(Result).filter(
        Result.user_id == user.id
    ).order_by(Result.created_at.desc()).all()

    return {
        "results": [
            {
                "id": r.id,
                "session_id": r.session_id,
                "predicted_field": r.predicted_field,
                "confidence_score": r.confidence_score,
                "all_scores": json.loads(r.all_scores) if r.all_scores else {},
                "explanation": r.explanation,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in results
        ]
    }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _get_traits(field: str) -> list:
    """Get personality traits for a given career field."""
    from backend.model.predictor import FIELD_DESCRIPTIONS
    return FIELD_DESCRIPTIONS.get(field, {}).get("traits", [])


def _get_skills(field: str) -> list:
    """Get suggested skills for a given career field."""
    from backend.model.predictor import FIELD_DESCRIPTIONS
    return FIELD_DESCRIPTIONS.get(field, {}).get("skills", [])
