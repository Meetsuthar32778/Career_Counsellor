# AI Career Counsellor: Architecture & Workflow Guide

This document is designed to help you confidently explain your project to your evaluators. It breaks down exactly how the system is built, the technologies used, and how data flows through the application.

---

## 🏗️ 1. High-Level Architecture

Your project is built using a modern **Client-Server Architecture** integrated with a **Machine Learning Pipeline**.

1. **Frontend (Presentation Layer):** HTML5, Vanilla CSS, and JavaScript. It handles the user interface, voice recognition (Web Speech API), dynamic chart rendering, and API communication.
2. **Backend (Application Layer):** Python with **FastAPI**. It manages routing, business logic, session state, and security (JWT authentication).
3. **Machine Learning Pipeline (AI Layer):** 
   - **NLP (Natural Language Processing):** `sentence-transformers` (all-MiniLM-L6-v2) to convert English text into mathematical vectors.
   - **Classifier:** `RandomForestClassifier` (Scikit-Learn) to predict the best career match based on those vectors.
4. **Database (Data Layer):** Relational database (SQLite/PostgreSQL) managed via **SQLAlchemy ORM** to store user credentials, chat histories, and final ML results.

---

## ⚙️ 2. Step-by-Step Workflow (Data Flow)

Here is exactly what happens from the moment a student opens the app to the moment they get their PDF report:

### Step 1: Authentication & Session Initialization
- The user logs in. FastAPI validates their credentials and issues a **JWT (JSON Web Token)** for secure API access.
- The user clicks "New Session". The frontend calls `/api/chat/start`.
- The backend creates a new `ChatSession` in the database and initializes the `AdaptiveQuestionEngine`.

### Step 2: Adaptive Chat & Live ML Tracking
- The bot asks a question (e.g., "What interests you most about automating tasks?").
- The user types a response (or uses the **Speech-to-Text** microphone).
- The text is sent to `/api/chat/message`. 
- **The Magic:** The `AdaptiveQuestionEngine` immediately passes the user's combined answers to the **Machine Learning Predictor**. The predictor returns real-time probabilities (e.g., CSE: 45%, Mech: 20%).
- The engine uses the highest scoring field to **adaptively select the next question** from that specific domain.
- The frontend receives this data and dynamically updates the **Live Personality Radar Chart** and the progress bar.

### Step 3: Final ML Prediction
- Once the 10th question is answered, the session is marked as `is_complete=True`.
- The frontend redirects the user to the `/result` page.
- The result page calls `/api/predict/result/{session_id}`.
- The backend fetches all 10 answers from the database, merges them into one large text block, and runs the **Final ML Pipeline**:
  1. **Embedding:** `SentenceTransformer` converts the text into a 384-dimensional mathematical array.
  2. **Classification:** The `RandomForest` model analyzes the array and outputs percentage probabilities for all 8 career fields.
  3. **Strength Detection:** Keyword heuristics scan the text to assign specific strengths (e.g., "Analytical Thinking").

### Step 4: Result Generation
- The backend saves the final percentages, strengths, and recommended skills to the `Result` database table.
- The frontend renders these beautifully using CSS animations, tags, and progress bars.
- The user can click "View Transcript" to see the database history of their chat, or click "Download PDF" (using `html2pdf.js`) to export the DOM into a professional report.

---

## 🛠️ 3. What Technology is Used for What?

When evaluators ask *why* you chose specific technologies, use these answers:

| Technology | Role in Project | Why was it chosen? |
| :--- | :--- | :--- |
| **FastAPI (Python)** | Backend Framework | It is incredibly fast, handles asynchronous requests perfectly (great for ML models), and auto-generates API documentation. |
| **Sentence-Transformers** | NLP Engine | Traditional keyword matching is weak. Transformers understand the *semantic context* (meaning) of sentences, making the AI highly intelligent. |
| **Random Forest (Scikit-Learn)**| ML Classifier | It handles high-dimensional data (like 384D text vectors) exceptionally well, prevents overfitting, and provides reliable probability scores for the Radar Chart. |
| **SQLAlchemy** | ORM (Database) | It allows Python to interact with the database using objects instead of raw SQL, making the code secure against SQL injection and easily portable. |
| **Web Speech API** | Accessibility | To provide a "Hybrid UX". Allowing students to dictate answers or hear the bot speak makes the app highly accessible and demo-ready. |
| **Vanilla CSS/JS** | Frontend UI | By avoiding heavy frameworks like React, the application remains lightweight, incredibly fast to load, and demonstrates strong fundamental web development skills. |

---

> [!TIP]
> **Viva Preparation Advice:** 
> Evaluators love "Explainable AI". If they ask how the AI works, emphasize that it doesn't just guess blindly. Explain that you use a **Two-Step Pipeline**: First, a Transformer turns text into math (Embeddings). Second, a Random Forest categorizes that math into Career Fields. This shows deep technical understanding!
