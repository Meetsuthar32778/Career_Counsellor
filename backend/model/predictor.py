"""
predictor.py - Career Prediction Engine
=========================================
Loads the pre-trained RandomForest model and sentence-transformer
to predict career fields from student responses with confidence scores.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths to saved model artifacts
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.txt")

# ---------------------------------------------------------------------------
# Career field metadata for explanations
# ---------------------------------------------------------------------------
FIELD_DESCRIPTIONS = {
    "CSE": {
        "full_name": "Computer Science & Engineering",
        "description": "Your responses indicate a strong affinity for logical thinking, problem-solving, and technology. You seem naturally drawn to computational thinking and digital innovation.",
        "traits": ["Analytical Thinking", "Problem Solving", "Logical Reasoning", "Innovation", "Technical Curiosity"],
        "skills": ["Programming & Software Development", "Data Structures & Algorithms", "Machine Learning & AI", "Web/App Development", "Database Management"],
    },
    "MECHANICAL": {
        "full_name": "Mechanical Engineering",
        "description": "Your answers reveal a passion for understanding how physical systems work. You show interest in mechanics, energy systems, and hands-on engineering design.",
        "traits": ["Spatial Reasoning", "Hands-on Aptitude", "Physics Intuition", "Design Thinking", "Practical Problem Solving"],
        "skills": ["CAD/CAM Design", "Thermodynamics", "Manufacturing Processes", "Robotics", "Automotive Engineering"],
    },
    "CIVIL": {
        "full_name": "Civil Engineering",
        "description": "Your responses suggest you care deeply about building things that last and improving infrastructure. You seem drawn to construction, planning, and structural design.",
        "traits": ["Structural Thinking", "Environmental Awareness", "Planning & Organization", "Visual-Spatial Skills", "Community Orientation"],
        "skills": ["Structural Analysis", "Urban Planning", "Surveying", "Construction Management", "Environmental Engineering"],
    },
    "ELECTRICAL": {
        "full_name": "Electrical Engineering",
        "description": "Your answers show a fascination with circuits, energy, and electronic systems. You demonstrate interest in how electrical systems power the modern world.",
        "traits": ["Analytical Precision", "Systems Thinking", "Mathematical Aptitude", "Circuit Intuition", "Technical Curiosity"],
        "skills": ["Circuit Design", "Power Systems", "Embedded Systems", "Signal Processing", "Renewable Energy"],
    },
    "BIOLOGY": {
        "full_name": "Biological Sciences",
        "description": "Your responses reveal a deep curiosity about living organisms and life processes. You show a natural interest in understanding health, nature, and biological systems.",
        "traits": ["Scientific Curiosity", "Observation Skills", "Empathy", "Research Orientation", "Detail Orientation"],
        "skills": ["Laboratory Techniques", "Genetics & Biotechnology", "Microbiology", "Ecology", "Biomedical Research"],
    },
    "CHEMISTRY": {
        "full_name": "Chemical Sciences",
        "description": "Your answers indicate a fascination with matter, reactions, and molecular structures. You seem drawn to understanding the composition of substances and how they interact.",
        "traits": ["Experimental Mindset", "Precision", "Analytical Thinking", "Scientific Rigor", "Curiosity about Materials"],
        "skills": ["Organic & Inorganic Chemistry", "Pharmaceutical Science", "Materials Science", "Analytical Chemistry", "Chemical Engineering"],
    },
    "PHYSICS": {
        "full_name": "Physical Sciences",
        "description": "Your responses show a deep interest in understanding the fundamental laws governing the universe. You demonstrate strong mathematical reasoning and abstract thinking.",
        "traits": ["Abstract Thinking", "Mathematical Excellence", "Theoretical Reasoning", "Curiosity about Nature", "Problem Solving"],
        "skills": ["Quantum Mechanics", "Astrophysics", "Nuclear Physics", "Computational Physics", "Research & Theory Development"],
    },
    "MANAGEMENT": {
        "full_name": "Business Administration (BBA/MBA)",
        "description": "Your answers reveal strong leadership qualities, communication skills, and business acumen. You seem naturally inclined towards organizing, leading, and strategic decision-making.",
        "traits": ["Leadership", "Communication", "Strategic Thinking", "Team Collaboration", "Decision Making"],
        "skills": ["Business Strategy", "Marketing & Sales", "Financial Management", "Human Resource Management", "Entrepreneurship"],
    },
}


class CareerPredictor:
    """
    Loads the pre-trained sentence-transformer and RandomForest model
    to predict career fields from aggregated student responses.
    """

    def __init__(self):
        """Initialize the predictor by loading the sentence-transformer and RF model."""
        print("[CareerPredictor] Loading sentence-transformer model...")
        self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[CareerPredictor] Sentence-transformer loaded successfully.")

        # Load the trained RandomForest classifier
        print(f"[CareerPredictor] Loading RF model from {RF_MODEL_PATH}...")
        with open(RF_MODEL_PATH, "rb") as f:
            self.rf_model = pickle.load(f)
        print("[CareerPredictor] RF model loaded successfully.")

        # Load the class labels
        with open(CLASSES_PATH, "r") as f:
            self.classes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"[CareerPredictor] Classes: {self.classes}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Convert a text string into a 384-dimensional embedding vector.

        Args:
            text: The input text (student answer).

        Returns:
            A numpy array of shape (384,).
        """
        return self.sentence_model.encode(text, show_progress_bar=False)

    def predict(self, answers: list) -> dict:
        """
        Predict the career field from a list of student answers.

        Workflow:
        1. Convert each answer into an embedding.
        2. Average all embeddings to get a single feature vector.
        3. Pass the feature vector to the RandomForest for prediction.
        4. Return the prediction, confidence, and detailed explanation.

        Args:
            answers: List of student answer strings.

        Returns:
            Dictionary with predicted_field, confidence_score, all_scores,
            explanation, strengths, personality_traits, and suggested_skills.
        """
        if not answers:
            return {
                "predicted_field": "UNKNOWN",
                "confidence_score": 0.0,
                "all_scores": {},
                "explanation": "No answers were provided.",
                "strengths": [],
                "personality_traits": [],
                "suggested_skills": [],
            }

        # Step 1: Compute embeddings for each answer
        embeddings = [self.get_embedding(answer) for answer in answers]

        # Step 2: Average the embeddings to create a single feature vector
        avg_embedding = np.mean(embeddings, axis=0).reshape(1, -1)

        # Step 3: Get prediction probabilities
        probabilities = self.rf_model.predict_proba(avg_embedding)[0]

        # Step 4: Map probabilities to class labels
        all_scores = {}
        for i, cls in enumerate(self.classes):
            all_scores[cls] = round(float(probabilities[i]) * 100, 2)

        # Step 5: Get the top prediction
        predicted_index = np.argmax(probabilities)
        predicted_field = self.classes[predicted_index]
        confidence_score = round(float(probabilities[predicted_index]) * 100, 2)

        # Step 6: Build the explanation from metadata
        field_info = FIELD_DESCRIPTIONS.get(predicted_field, {})

        return {
            "predicted_field": predicted_field,
            "full_name": field_info.get("full_name", predicted_field),
            "confidence_score": confidence_score,
            "all_scores": all_scores,
            "explanation": field_info.get("description", "Based on your responses, this field aligns well with your interests."),
            "strengths": self._detect_strengths(answers),
            "personality_traits": field_info.get("traits", []),
            "suggested_skills": field_info.get("skills", []),
        }

    def _detect_strengths(self, answers: list) -> list:
        """
        Analyze all answers to detect broad strength categories.

        Args:
            answers: List of student answer strings.

        Returns:
            List of detected strength labels.
        """
        combined = " ".join(answers).lower()
        strengths = []

        # Keyword-based strength detection
        strength_keywords = {
            "Analytical Thinking": ["analyze", "logic", "data", "problem", "solve", "algorithm", "calculate", "reason"],
            "Creative Thinking": ["creative", "design", "imagine", "innovate", "art", "build", "invent", "idea"],
            "Leadership": ["lead", "team", "manage", "organize", "delegate", "motivate", "inspire", "coordinate"],
            "Communication": ["communicate", "present", "write", "speak", "negotiate", "persuade", "discuss", "express"],
            "Scientific Curiosity": ["experiment", "research", "discover", "curious", "explore", "investigate", "observe", "hypothesis"],
            "Technical Skills": ["code", "program", "software", "hardware", "circuit", "machine", "tool", "technology"],
            "Mathematical Aptitude": ["math", "equation", "number", "formula", "calculus", "geometry", "statistics", "algebra"],
            "Environmental Awareness": ["environment", "nature", "sustain", "ecology", "green", "conservation", "climate", "pollution"],
        }

        for strength, keywords in strength_keywords.items():
            if any(kw in combined for kw in keywords):
                strengths.append(strength)

        # Always return at least a couple of generic strengths
        if len(strengths) < 2:
            strengths.extend(["Dedication & Hard Work", "Willingness to Learn"])

        return strengths[:6]  # Cap at 6 strengths


# ---------------------------------------------------------------------------
# Singleton instance (loaded once when module is imported)
# ---------------------------------------------------------------------------
_predictor_instance = None


def get_predictor() -> CareerPredictor:
    """
    Returns the singleton CareerPredictor instance.
    Loads the model lazily on first call.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CareerPredictor()
    return _predictor_instance
