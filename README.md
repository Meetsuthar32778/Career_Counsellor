# 🎓 AI Based Intelligent Career Counsellor Chatbot

> An intelligent chatbot that uses **indirect questioning** and **NLP-based semantic analysis** to discover a student's latent interests and recommend the best career path.

## 📋 Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [AI Workflow](#ai-workflow)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [Deployment](#deployment)
- [Features](#features)
- [Viva Explanation](#viva-explanation)
- [Future Scope](#future-scope)

---

## 🎯 Overview

This project implements an **AI-powered Career Counsellor Chatbot** designed for university admissions. Unlike traditional rule-based bots, this system:

- Acts as a **human interviewer** using **indirect contextual questions**
- Analyzes **open-ended text responses** using NLP
- **Adapts questions dynamically** based on detected interests
- Predicts the best career path with **confidence scores**
- Generates a **detailed explanation report**

### Supported Career Paths

| Category | Fields |
|----------|--------|
| **Engineering** | Computer Science, Mechanical, Civil, Electrical |
| **Pure Sciences** | Biology, Chemistry, Physics |
| **Management** | BBA / MBA |

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| Backend | Python FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite + SQLAlchemy ORM |
| NLP Model | sentence-transformers/all-MiniLM-L6-v2 |
| ML Classifier | RandomForestClassifier (scikit-learn) |
| Authentication | JWT (python-jose) + bcrypt |
| Speech Output | Web Speech API |
| PDF Export | html2pdf.js |

---

## 🧠 AI Workflow

```
Student Answers → Sentence-Transformer → 384-dim Embeddings → Average Pooling → RandomForest → Prediction + Confidence
```

1. **Data Collection**: Student answers are collected through indirect questions
2. **Text Embedding**: Each answer is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`
3. **Feature Aggregation**: All embeddings are averaged into a single feature vector
4. **Classification**: The RandomForestClassifier predicts the career field
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
