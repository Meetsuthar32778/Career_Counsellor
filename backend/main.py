"""
main.py - FastAPI Application Entry Point
===========================================
Initializes the FastAPI application, configures middleware,
includes all API routers, and sets up static file serving.

AI Career Counsellor Chatbot
============================
An intelligent chatbot that uses indirect questioning to discover
a student's latent interests and recommend a career path in
Engineering, Science, or Management.

Tech Stack:
- Backend: FastAPI + SQLAlchemy + SQLite
- AI/ML: sentence-transformers + scikit-learn (RandomForest)
- Frontend: HTML + CSS + JavaScript + Bootstrap
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os

# Import database initialization
from .database.database import init_db

# Import API routers
from .routes.auth_routes import router as auth_router
from .routes.chat_routes import router as chat_router
from .routes.prediction_routes import router as prediction_router

# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Career Counsellor Chatbot",
    description=(
        "An intelligent career counselling chatbot that uses indirect questioning "
        "and NLP-based analysis to recommend career paths for students."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS Middleware (allow all origins for development)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static Files & Templates Directory
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BACKEND_DIR, "static")
TEMPLATES_DIR = os.path.join(BACKEND_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _read_template(filename: str) -> str:
    """Read an HTML template file and return its contents."""
    filepath = os.path.join(TEMPLATES_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Include API Routers
# ---------------------------------------------------------------------------
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(prediction_router)

# ---------------------------------------------------------------------------
# Database Initialization (runs on startup)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    """Initialize the database tables on application startup."""
    print("=" * 60)
    print("  AI Career Counsellor Chatbot - Starting Up...")
    print("=" * 60)
    init_db()
    print("[Database] Tables created/verified successfully.")
    print("[Server] Application is ready!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Page Routes (Serve HTML Templates)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_auth_page():
    """
    Serve the Login/Register page.
    This is the landing page for unauthenticated users.
    """
    return HTMLResponse(content=_read_template("auth.html"))


@app.get("/chat", response_class=HTMLResponse)
async def serve_chat_page():
    """
    Serve the main Chat interface page.
    Users interact with the AI counsellor here.
    """
    return HTMLResponse(content=_read_template("index.html"))


@app.get("/result/{session_id}", response_class=HTMLResponse)
async def serve_result_page(session_id: int):
    """
    Serve the Result/Recommendation page.
    Displays the final career prediction and analysis.
    """
    return HTMLResponse(content=_read_template("result.html"))


# ---------------------------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Simple health check endpoint for monitoring."""
    return {"status": "healthy", "service": "AI Career Counsellor Chatbot"}
