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
- **Feedback Loop:** Implement a user rating system on the final results page to collect active feedback, enabling continuous online fine-tuning of the Random Forest model over time.
