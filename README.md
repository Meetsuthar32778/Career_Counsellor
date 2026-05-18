# AI Career Counsellor Chatbot

An intelligent, AI-powered career counselling web application that discovers a student's latent interests through indirect, adaptive questioning, recommending paths across Engineering, Science, or Management.

---

## 1. Project Overview

The **AI Career Counsellor Chatbot** is a modern web application designed to act as a virtual career counselor. Instead of asking direct, generic questions (like "Do you like math?"), the application engages the user in a fluid chat, adaptively asking scenario-based questions. 

**Key Features:**
- **Live ML Analysis:** As the user answers questions, their text is encoded and processed by a Machine Learning pipeline in the background to detect interests.
- **Adaptive Questioning:** The chat engine dynamically selects the next question based on the user's responses, progressively narrowing down the most suitable career fields.
- **Dynamic Frontend Updates:** A live radar chart and progress bar update in real-time without interrupting the chat experience.
- **Comprehensive Reports:** At the end of the session, users receive a detailed analysis page, complete with their predicted career field, personality traits, suggested skills to develop, and an option to download a final PDF report.

---

## 2. System Architecture

```text
       +--------------------+
       |    User Browser    |
       |  (HTML, CSS, JS)   |
       +---------+----------+
                 | HTTP / REST API
                 v
       +---------+----------+
       |   FastAPI Backend  | 
       | (Routers, Services)|
       +----+----------+----+
            |          |
      Async |          | ThreadPool
            v          v
 +----------+---+ +----+-------------+
 | SQLite DB    | |   ML Pipeline    |
 | (SQLAlchemy) | |(RandomForest &   |
 | (Users,      | | Sentence-        |
 |  Sessions,   | | Transformers)    |
 |  Results)    | +------------------+
 +--------------+ 
```

**Frontend (HTML/CSS/JS):** A custom, responsive UI (dark mode SaaS aesthetic) that handles chat interactions, live charting (using Chart.js), and PDF generation (using jsPDF).
**Backend (FastAPI):** A fast, asynchronous backend handling chat logic, session management, database transactions, and dispatching ML inference to background threads so the event loop remains unblocked.
**Database:** SQLite via SQLAlchemy ORM, robustly tracking users, chat transcripts, and final ML results.
**ML Engine:** Uses Hugging Face's `sentence-transformers` for embedding extraction, coupled with a highly tuned `RandomForestClassifier` for classification.

---

## 3. File Structure & Responsibilities

| File/Folder | Purpose | Dependencies / Key Roles |
| :--- | :--- | :--- |
| **`main.py`** | Application Entry Point | Sets up FastAPI, mounts static routes, initializes the SQLite database, and includes API routers. |
| **`routes/chat_routes.py`** | Chat API Endpoints | Handles `/api/chat/message`. Offloads ML processing to `fastapi.concurrency.run_in_threadpool`. |
| **`routes/prediction_routes.py`** | Final Prediction API | Handles `/api/predict/result/{session_id}`. Fetches all answers, calls `predict()` asynchronously, and saves the final result. |
| **`routes/auth_routes.py`** | Authentication API | Handles login and registration endpoints. |
| **`services/chat_engine.py`** | Adaptive Questioning Engine | Contains `AdaptiveQuestionEngine`. Decides which question to ask next, tracks live field scores, and calls ML via threadpool. |
| **`model/predictor.py`** | ML Inference Wrapper | Contains `CareerPredictor`. Caches ML responses in memory, tokenizes input, and returns predictions and confidence scores. |
| **`model/model_training.py`** | ML Training Script | Extracts dataset, runs SMOTE for class balance, tunes hyperparameters via `RandomizedSearchCV`, and saves the final model. |
| **`model/evaluate_model.py`** | ML Evaluation Script | Used to measure the model's accuracy, F1-score, and confusion matrix against holdout data. |
| **`model/preprocessing.py`** | Data Cleaning | Contains `clean_text()` to normalize input (lowercase, remove punctuation/stopwords) for both training and inference. |
| **`database/database.py`** | DB Connection | Sets up the SQLite engine and `SessionLocal` factory. |
| **`database/models.py`** | DB Schema Models | SQLAlchemy models for `User`, `ChatSession`, `ChatHistory`, and `Result`. |
| **`database/schemas.py`** | Pydantic Schemas | Data validation for incoming requests and outgoing API responses (e.g., `PredictionResponse`). |
| **`templates/index.html`** | Chat UI Template | The main interactive chatbot UI featuring real-time radar charts and typing indicators. |
| **`templates/result.html`** | Recommendation UI | Displays the final career prediction, skills, and traits, and handles PDF downloads. |
| **`templates/auth.html`** | Login UI | Responsive landing page for user sign-in and registration. |
| **`static/js/main.js`** | Chat Logic (Frontend) | Handles AJAX requests to the backend, rendering chat bubbles, and updating Chart.js in real time. |
| **`static/css/style.css`** | Global Stylesheet | Modern dark-mode styling, variables, glassmorphism, and responsive breakpoints. |

---

## 4. Complete Workflow (Step by Step)

1. **Authentication:** The user arrives at `/` (`auth.html`). They register or log in. The backend issues a simple JWT or session token.
2. **Session Initialization:** Upon redirecting to `/chat` (`index.html`), the frontend requests a new chat session. The backend creates a `ChatSession` record.
3. **Adaptive Interaction:** The bot asks the first question. The user responds.
4. **Backend Processing:** `chat_routes.py` captures the answer and passes it to `AdaptiveQuestionEngine`.
5. **Live ML Update:** The engine excludes demographic answers, concatenates the text, and calls `CareerPredictor.predict()` inside a background thread pool. It recalculates the probabilities for the various career fields and selects the next question dynamically.
6. **Frontend Update:** The radar chart animates to reflect the updated ML probability scores in real-time.
7. **Finalization:** After a set number of questions, the chat concludes. The frontend is notified and redirects to `/result/{session_id}` (`result.html`).
8. **Final Prediction:** `prediction_routes.py` aggregates all chat answers, re-runs the ML pipeline for maximum context, and generates the final field recommendation, skills, and personality traits.
9. **PDF Report:** The user can click "Download Report (PDF)" on the result page, which triggers `jsPDF` to compile the on-screen data into a downloadable document.

---

## 5. ML Pipeline Deep Dive

### Data Preprocessing
The `preprocessing.py` script houses the `clean_text()` function. It converts text to lowercase, strips punctuation, and removes common English stop words. This is critical to reduce noise and standardize input tokens.

### Text Embedding Strategy
We utilize the `sentence-transformers` library, specifically the **`all-MiniLM-L6-v2`** model. This is an extremely fast, high-quality, lightweight model optimized for semantic similarity.
- **Important Change - Concatenation vs Averaging:** Previously, the system generated separate embeddings for each chat response and averaged them. Averaging embeddings dilutes semantic meaning (e.g., mixing "math" and "art" results in a generic middle ground). We updated the pipeline to **concatenate** all of the user's answers into a single string *before* generating the embedding, perfectly preserving the contextual richness of the entire conversation.

### Classifier: Random Forest
A `RandomForestClassifier` is used on top of the embeddings.
- **Why?** It handles high-dimensional embedding data robustly, naturally prevents overfitting (compared to deep neural networks on small datasets), captures non-linear relationships, and provides out-of-the-box probability distributions (`predict_proba`) which drive our live radar chart.

### Consistency & State Management
- Both training (`model_training.py`) and inference (`predictor.py`) strictly enforce the exact same `clean_text()` pipeline.
- The **first demographic question** (e.g., "What is your current grade/degree?") is explicitly excluded from the ML embedding input using `skip_first=True`. This prevents the model from developing biases based strictly on academic year rather than latent interests.

---

## 6. Hyperparameters Used & Rationale

### ML Pipeline Parameters (`model_training.py`)
- **`n_estimators`**: Tuned to values like 100, 200, 300, 500 via RandomizedSearchCV. Selected `300` generally for the optimal balance of accuracy and speed.
- **`max_depth`**: Tuned (10, 20, None). Constraining depth prevents overfitting on limited datasets.
- **`min_samples_split` / `min_samples_leaf`**: Tuning these parameters acts as regularization for the Random Forest trees.
- **`class_weight='balanced'`**: Instructs the Random Forest to automatically adjust weights inversely proportional to class frequencies, crucial for imbalanced career datasets.
- **`SMOTE(random_state=42)`**: Synthetic Minority Over-sampling Technique. `k_neighbors` defaults to 5. It artificially generates examples for minority career paths so the model isn't biased toward over-represented careers.
- **SentenceTransformer**: `batch_size=32` is used under the hood. `show_progress_bar=True` is enabled during training to track bulk encoding.

### Application Caching Parameters (`predictor.py`)
- **LRU Cache (Dict)**: The ML predictor maintains a dictionary where the key is a hashable `tuple` of the cleaned user answers. Because the live chat evaluates answers incrementally, caching previous embedding calculations significantly reduces CPU overhead.

---

## 7. Error Handling & Edge Cases

- **Missing/Short Answers:** The ML model handles short strings via the fallback behavior of the embedding model. If no text is provided, `predictor.py` immediately returns a safe "UNKNOWN" fallback dictionary.
- **Non-blocking Architecture:** If the ML pipeline takes a moment to process large embeddings, `fastapi.concurrency.run_in_threadpool` prevents the ASGI server from hanging, meaning other users' requests will continue to be processed instantly.
- **Low Confidence Prediction:** If `predict_proba` returns a maximum confidence score below **60%**, the backend flags `low_confidence_warning = True`. The frontend handles this by rendering a red warning banner ("Our confidence in this recommendation is moderate. Please consider taking additional assessments.") above the final results.

---

## 8. Setup & Installation

### Prerequisites
- Python 3.9+
- `pip` package manager

### Installation Steps
1. **Clone the repository.**
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Train the ML Model:** (Required if the joblib files do not exist)
   ```bash
   cd backend/model
   python model_training.py
   ```
5. **Start the Backend Server:**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```
6. **Access the App:** Open your browser and navigate to `http://127.0.0.1:8000`.

---

## 9. Testing & Validation

### Model Evaluation
To evaluate the strict performance of the model on the current dataset without running the web server:
```bash
cd backend/model
python evaluate_model.py
```
This script will output a detailed `classification_report` (Precision, Recall, F1-Score) and a `confusion_matrix` indicating where the model might be confusing related fields.

### Manual UI Testing
- Open the chat UI, interact with the bot, and observe the live radar chart updates.
- Test edge cases by inputting garbage text or extremely short answers to verify that the `low_confidence_warning` properly triggers on the results page.

---

## 10. Future Improvements

- **Stateless Backend:** Currently, the `AdaptiveQuestionEngine` maintains state in memory (`active_engines` dictionary). For horizontal scalability (e.g., Kubernetes, multiple Uvicorn workers), this state should be migrated to a Redis cache.
- **LLM Integration (Agentic Approach):** Transition the decision-making engine to an LLM (using LangChain or LangGraph) instead of rule-based keyword matching to handle vastly nuanced and unpredictable user feedback dynamically.
- **Feedback Loop:** Implement a user rating system on the final results page to collect active feedback, enabling continuous online fine-tuning of the Random Forest model over time.4. **Classification**: The RandomForestClassifier predicts the career field
5. **Confidence Scoring**: Prediction probabilities give confidence percentages
6. **Explanation Generation**: Strengths, traits, and skills are mapped from metadata

### Adaptive Questioning Engine

The chatbot uses a **dynamic question selection strategy**:

- **First 3 questions**: Mix of general and broad exploration
- **60% of remaining**: Dive deeper into the highest-scoring field
- **20%**: Explore least-covered fields (breadth)
- **20%**: General probing questions

Keywords in answers are matched against field-specific dictionaries to update real-time confidence scores.

---

## 📁 Project Structure

```
Career_Counsellor/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── database/
│   │   ├── database.py          # SQLAlchemy setup
│   │   ├── models.py            # ORM models (users, sessions, etc.)
│   │   └── schemas.py           # Pydantic validation schemas
│   ├── routes/
│   │   ├── auth_routes.py       # Login/Register APIs
│   │   ├── chat_routes.py       # Chat session APIs
│   │   └── prediction_routes.py # Prediction APIs
│   ├── services/
│   │   └── chat_engine.py       # Adaptive Questioning Engine
│   ├── model/
│   │   ├── predictor.py         # ML prediction pipeline
│   │   ├── rf_model.pkl         # Trained RandomForest model
│   │   └── classes.txt          # Class labels
│   ├── dataset/
│   │   └── career_dataset.csv   # Training dataset (1000+ samples)
│   ├── static/
│   │   ├── css/style.css        # Premium dark mode styling
│   │   └── js/main.js           # Frontend chat logic
│   ├── templates/
│   │   ├── auth.html            # Login/Register page
│   │   ├── index.html           # Chat interface
│   │   └── result.html          # Result/Recommendation page
│   └── utils/
│       └── security.py          # JWT + password hashing
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone or navigate to the project directory
cd Career_Counsellor

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Application

```bash
# Start the FastAPI server
uvicorn backend.main:app --reload

# Open in browser
# http://127.0.0.1:8000
```

1. **Register** a new account
2. **Login** with your credentials
3. Click **"+ New Session"** to start
4. Answer 10 indirect questions honestly
5. View your **personalized career recommendation**

---

## 🧪 Training & Model Comparison

```bash
# Train RandomForest model used by the app
python -m backend.model.model_training

# Compare Random Forest vs Linear SVM vs Logistic Regression
python -m backend.model.compair_model
```

The comparison script prints fold-wise metrics and saves a ranked summary to:
`backend/model/comparison_results.csv`

---

## 🌐 Deployment

### Render / Railway

1. Push the project to GitHub
2. Connect the repository on Render/Railway
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

---

## ✨ Features

- ✅ Indirect contextual questioning (never asks direct career questions)
- ✅ Adaptive question selection based on previous answers
- ✅ NLP-powered semantic analysis (sentence-transformers)
- ✅ RandomForest ML classification with confidence scores
- ✅ Modern dark-mode ChatGPT-like UI
- ✅ Speech output (Text-to-Speech)
- ✅ PDF report download
- ✅ User authentication (JWT)
- ✅ Chat history storage (SQLite)
- ✅ Progress tracking with visual progress bar
- ✅ Animated typing indicator
- ✅ Responsive design (mobile-friendly)

---

## 🎤 Viva Explanation

### Q: How does the chatbot decide what question to ask next?
**A:** The Adaptive Questioning Engine maintains real-time confidence scores for each career field. After each answer, it analyzes keywords and updates scores. The next question is selected from the field with the highest emerging interest (60% of the time), or from under-explored fields to maintain breadth (40% of the time).

### Q: Why indirect questions instead of direct?
**A:** Direct questions lead to biased responses. Indirect questions like "If you had a free day, how would you spend it?" reveal genuine interests without leading the student toward any particular field.

### Q: How does the ML model work?
**A:** Each student answer is converted to a 384-dimensional embedding using the `all-MiniLM-L6-v2` sentence transformer. All embeddings are averaged into a single feature vector, which is fed into a trained RandomForestClassifier to predict the career field with probability scores.

### Q: What is sentence-transformers?
**A:** It's a framework for generating dense vector representations (embeddings) of text. The `all-MiniLM-L6-v2` model maps sentences to a 384-dimensional space where semantically similar sentences are close together.

---

## 🔮 Future Scope

- Integration with larger LLMs (GPT, Gemini) for more natural conversations
- Multi-language support
- Detailed psychometric analysis
- Integration with university admission systems
- Mobile app development
- Recommendation of specific colleges/courses
