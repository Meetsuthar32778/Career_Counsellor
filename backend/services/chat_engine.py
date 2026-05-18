"""
chat_engine.py - Adaptive Contextual Questioning Engine
========================================================
The core intelligence of the Career Counsellor Chatbot.

This engine:
1. Maintains a pool of INDIRECT questions categorized by career field.
2. Analyzes user responses using keyword/semantic matching.
3. Dynamically adjusts career confidence scores based on answers.
4. Selects the NEXT question adaptively based on detected interests.
5. Never asks direct questions - only indirect, contextual ones.
"""

import random
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Total number of questions in a counselling session
# ---------------------------------------------------------------------------
TOTAL_QUESTIONS = 10

# ---------------------------------------------------------------------------
# QUESTION BANK: Indirect questions organized by target career field.
# Each question is designed to probe latent interests WITHOUT directly
# mentioning the career field.
# ---------------------------------------------------------------------------
QUESTION_BANK = {
    # ======================== GENERAL / OPENING QUESTIONS ========================
    "GENERAL": [
        {
            "question": "If you had a completely free day, how would you choose to spend it?",
            "keywords": [],
            "follow_up_field": None,
            "suggestions": ["💡 Playing video games", "💡 Building/fixing things", "💡 Reading books", "💡 Hanging out with friends"]
        },
        {
            "question": "What is a topic you can talk about for hours without getting bored?",
            "keywords": [],
            "follow_up_field": None,
            "suggestions": ["💡 Technology & gadgets", "💡 Space & universe", "💡 Business & startups", "💡 Nature & animals"]
        },
        {
            "question": "If you could solve one big world problem, what approach would you take?",
            "keywords": [],
            "follow_up_field": None,
            "suggestions": ["💡 Climate change", "💡 Cybersecurity", "💡 Curing diseases", "💡 Poverty & economy"]
        },
        {
            "question": "What kind of YouTube videos or documentaries naturally grab your attention?",
            "keywords": [],
            "follow_up_field": None,
            "suggestions": ["💡 Tech reviews", "💡 Science experiments", "💡 Architecture/Mega-structures", "💡 Business case studies"]
        },
    ],

    # ======================== CSE-ORIENTED QUESTIONS ========================
    "CSE": [
        {
            "question": "What interests you most about automating tasks or how apps work behind the scenes?",
            "keywords": ["automate", "code", "script", "program", "computer", "software", "app", "bot", "algorithm"],
            "suggestions": ["💡 The logic and algorithms", "💡 Building artificial intelligence", "💡 Writing code", "💡 I prefer other fields"]
        },
        {
            "question": "How would you describe your experience with creating websites, games, or digital tools?",
            "keywords": ["game", "website", "tool", "build", "create", "code", "develop", "hack"],
            "suggestions": ["💡 Built a simple website", "💡 Tried game modding", "💡 Wrote python scripts", "💡 Haven't tried but want to"]
        },
        {
            "question": "If you could teach a robot to do one thing perfectly, what would it be and why?",
            "keywords": ["robot", "artificial", "intelligence", "AI", "machine", "learn", "train", "automate"],
            "suggestions": ["💡 Write its own software", "💡 Build mechanical parts", "💡 Manage schedules", "💡 Perform surgeries"]
        },
        {
            "question": "What is your favorite strategy when solving logic puzzles, brain teasers, or games like chess?",
            "keywords": ["puzzle", "logic", "brain", "pattern", "solve", "riddle", "chess", "sudoku", "strategy"],
            "suggestions": ["💡 Finding mathematical patterns", "💡 Planning multiple steps ahead", "💡 Using pure logic", "💡 I prefer physical activities"]
        },
    ],

    # ======================== MECHANICAL-ORIENTED QUESTIONS ========================
    "MECHANICAL": [
        {
            "question": "What fascinates you the most when you take apart a gadget or machine to see how it works?",
            "keywords": ["machine", "engine", "gear", "motor", "disassemble", "mechanism", "parts", "mechanical"],
            "suggestions": ["💡 The physical gears and motors", "💡 How the engine produces power", "💡 I prefer looking at the software", "💡 The external design"]
        },
        {
            "question": "How would you design a vehicle for extreme terrain? What features would be crucial?",
            "keywords": ["vehicle", "engine", "wheel", "suspension", "design", "terrain", "power", "torque", "aerodynamic"],
            "suggestions": ["💡 High engine power & torque", "💡 AI autopilot software", "💡 Strong physical suspension", "💡 Solar energy battery"]
        },
        {
            "question": "What catches your attention when you see heavy machinery like cranes or industrial robots?",
            "keywords": ["crane", "lift", "force", "weight", "hydraulic", "pulley", "mechanical", "energy", "torque"],
            "suggestions": ["💡 The mechanical lifting power", "💡 The physics behind the pulleys", "💡 The software controlling it", "💡 The manufacturing process"]
        },
    ],

    # ======================== CIVIL-ORIENTED QUESTIONS ========================
    "CIVIL": [
        {
            "question": "When you visit a new city, what aspects of the buildings, roads, or layout stand out to you?",
            "keywords": ["building", "road", "bridge", "structure", "city", "architecture", "layout", "plan", "infrastructure"],
            "suggestions": ["💡 The skyscraper architecture", "💡 The road and transit infrastructure", "💡 I notice nature more", "💡 I notice the businesses"]
        },
        {
            "question": "How do you think engineers design buildings to survive earthquakes while others fall?",
            "keywords": ["earthquake", "structure", "foundation", "concrete", "steel", "design", "resistant", "collapse", "construct"],
            "suggestions": ["💡 Strong concrete foundations", "💡 Flexible steel structures", "💡 Good architectural planning", "💡 Advanced physical calculations"]
        },
        {
            "question": "What part of building massive bridges or dams interests you the most?",
            "keywords": ["bridge", "dam", "construct", "foundation", "span", "load", "soil", "concrete", "engineering"],
            "suggestions": ["💡 The structural engineering", "💡 The raw materials used", "💡 Planning the city layout", "💡 I'm not really interested"]
        },
    ],

    # ======================== ELECTRICAL-ORIENTED QUESTIONS ========================
    "ELECTRICAL": [
        {
            "question": "What do you wonder about when you think of how electricity reaches your home or charges your phone?",
            "keywords": ["electricity", "circuit", "wire", "voltage", "current", "power", "charger", "transformer", "grid"],
            "suggestions": ["💡 How circuits and voltage work", "💡 How massive power grids function", "💡 I focus more on the software", "💡 I prefer mechanical engines"]
        },
        {
            "question": "How would you design a new renewable energy device, and what source would it use?",
            "keywords": ["solar", "wind", "energy", "battery", "generator", "renewable", "power", "electric", "panel"],
            "suggestions": ["💡 Advanced solar panels", "💡 Efficient wind turbines", "💡 Better battery technology", "💡 Kinetic energy systems"]
        },
        {
            "question": "What interests you about how sensors, motors, or LEDs function on a circuit board?",
            "keywords": ["sensor", "motor", "component", "electronic", "microcontroller", "circuit", "LED", "resistor", "capacitor"],
            "suggestions": ["💡 Tinkering with microcontrollers", "💡 Understanding electronic sensors", "💡 I prefer purely coding", "💡 I'm not interested"]
        },
    ],

    # ======================== BIOLOGY-ORIENTED QUESTIONS ========================
    "BIOLOGY": [
        {
            "question": "If you could look at anything under a powerful microscope, what would you choose and why?",
            "keywords": ["cell", "bacteria", "microscope", "organism", "DNA", "tissue", "blood", "microbe", "virus"],
            "suggestions": ["💡 Human cells or DNA", "💡 Plant structures", "💡 Computer microchips", "💡 Chemical reactions"]
        },
        {
            "question": "What do you find most fascinating about how the human body heals itself?",
            "keywords": ["heal", "body", "immune", "cell", "blood", "regenerate", "medicine", "health", "biology"],
            "suggestions": ["💡 The immune system", "💡 Cellular regeneration", "💡 Medical treatments", "💡 It's all chemistry to me"]
        },
        {
            "question": "What are your thoughts on the possibilities and ethics of genetic engineering?",
            "keywords": ["genetic", "DNA", "clone", "gene", "modify", "biotechnology", "ethics", "genome", "hereditary"],
            "suggestions": ["💡 Biotechnology is the future", "💡 DNA editing is fascinating", "💡 It requires careful management", "💡 Prefer not to say"]
        },
    ],

    # ======================== CHEMISTRY-ORIENTED QUESTIONS ========================
    "CHEMISTRY": [
        {
            "question": "What interests you about working in a laboratory and mixing chemical substances?",
            "keywords": ["mix", "reaction", "chemical", "substance", "experiment", "compound", "solution", "acid", "base"],
            "suggestions": ["💡 Lab experiments are fun", "💡 I like complex chemical reactions", "💡 I prefer computer labs", "💡 I prefer working outdoors"]
        },
        {
            "question": "How do you think medicines interact with our bodies at a molecular level?",
            "keywords": ["medicine", "drug", "pharmaceutical", "molecule", "receptor", "dose", "formula", "organic", "chemistry"],
            "suggestions": ["💡 Organic chemistry", "💡 Pharmaceutical development", "💡 Molecular bonding", "💡 The body's biological response"]
        },
        {
            "question": "If you could invent a new material, what properties would you give it?",
            "keywords": ["material", "polymer", "alloy", "compound", "crystal", "property", "synthetic", "lab", "chemical"],
            "suggestions": ["💡 A super-light synthetic polymer", "💡 An unbreakable metallic alloy", "💡 A self-healing biological material", "💡 A highly conductive electronic material"]
        },
    ],

    # ======================== PHYSICS-ORIENTED QUESTIONS ========================
    "PHYSICS": [
        {
            "question": "When you watch a roller coaster, what physical forces or mechanics catch your attention?",
            "keywords": ["force", "gravity", "motion", "speed", "acceleration", "energy", "momentum", "velocity", "friction"],
            "suggestions": ["💡 Velocity and gravity", "💡 The mechanical structure", "💡 I just enjoy the ride", "💡 The business cost of building it"]
        },
        {
            "question": "What theories come to mind when you look at the night sky and think about black holes?",
            "keywords": ["star", "universe", "galaxy", "black hole", "space", "light", "cosmos", "astronomy", "quantum"],
            "suggestions": ["💡 The theory of relativity", "💡 Quantum mechanics", "💡 How to build a spaceship", "💡 Extraterrestrial biology"]
        },
        {
            "question": "How would you explain why the sky is blue using scientific principles?",
            "keywords": ["light", "scatter", "wavelength", "color", "spectrum", "optics", "refraction", "wave", "photon"],
            "suggestions": ["💡 Light wavelength scattering", "💡 The electromagnetic spectrum", "💡 Chemical composition of air", "💡 I have no idea"]
        },
    ],

    # ======================== MANAGEMENT-ORIENTED QUESTIONS ========================
    "MANAGEMENT": [
        {
            "question": "How would you organize and plan a massive college festival?",
            "keywords": ["organize", "plan", "team", "budget", "manage", "lead", "coordinate", "event", "delegate"],
            "suggestions": ["💡 Delegate tasks to a team", "💡 Focus on the budget and finance", "💡 Build an app for coordination", "💡 Design the main stage structure"]
        },
        {
            "question": "What is your approach when trying to convince a group of people to agree on a strategy?",
            "keywords": ["convince", "negotiate", "persuade", "team", "lead", "communicate", "agreement", "influence", "strategy"],
            "suggestions": ["💡 Using strong communication skills", "💡 Negotiating a compromise", "💡 Showing them data and logic", "💡 I prefer working alone"]
        },
        {
            "question": "If you were to start a small business tomorrow, what would it be and how would you manage it?",
            "keywords": ["business", "startup", "profit", "market", "customer", "sell", "invest", "entrepreneur", "revenue"],
            "suggestions": ["💡 A tech startup", "💡 A marketing agency", "💡 An engineering consultancy", "💡 A pharmaceutical company"]
        },
    ],
}


# ---------------------------------------------------------------------------
# Keyword weights for dynamic scoring
# ---------------------------------------------------------------------------
FIELD_KEYWORD_MAP = {
    "CSE": ["code", "program", "software", "algorithm", "data", "computer", "app", "website", "automate",
            "AI", "artificial", "machine learning", "debug", "hack", "tech", "digital", "cyber", "python",
            "java", "database", "server", "cloud", "logic", "binary", "internet", "network", "compile"],
    "MECHANICAL": ["machine", "engine", "gear", "motor", "thermal", "heat", "automobile", "robot",
                   "manufacture", "design", "CAD", "turbine", "piston", "fluid", "dynamics", "torque",
                   "force", "mechanic", "vehicle", "aerospace", "friction", "tool", "workshop"],
    "CIVIL": ["building", "bridge", "road", "structure", "construct", "concrete", "soil", "foundation",
              "architecture", "urban", "plan", "infrastructure", "dam", "survey", "environment",
              "water supply", "drainage", "earthquake", "steel", "cement", "city", "housing"],
    "ELECTRICAL": ["circuit", "voltage", "current", "power", "electric", "wire", "transformer", "battery",
                   "solar", "energy", "signal", "frequency", "motor", "generator", "grid", "semiconductor",
                   "diode", "transistor", "sensor", "microcontroller", "renewable", "LED", "electromagnetic"],
    "BIOLOGY": ["cell", "DNA", "gene", "organism", "species", "evolution", "body", "health", "medicine",
                "bacteria", "virus", "ecology", "plant", "animal", "microscope", "biotechnology",
                "anatomy", "physiology", "immune", "blood", "protein", "enzyme", "tissue", "life"],
    "CHEMISTRY": ["chemical", "reaction", "molecule", "atom", "compound", "element", "acid", "base",
                  "organic", "bond", "solution", "lab", "experiment", "polymer", "catalyst", "oxidation",
                  "pharmaceutical", "formula", "periodic", "titration", "substance", "crystal", "alloy"],
    "PHYSICS": ["force", "gravity", "motion", "energy", "quantum", "relativity", "wave", "light",
                "particle", "atom", "nuclear", "electromagnetic", "velocity", "acceleration", "mass",
                "newton", "einstein", "thermodynamics", "optics", "spectrum", "cosmos", "universe"],
    "MANAGEMENT": ["lead", "team", "manage", "business", "strategy", "market", "finance", "organize",
                   "plan", "entrepreneur", "profit", "invest", "negotiate", "communicate", "HR",
                   "brand", "customer", "revenue", "startup", "budget", "delegate", "motivate", "vision"],
}


# ============================================================
# Conversational Memory Phrases
# ============================================================
MEMORY_PHRASES = {
    "CSE": ["Since you showed an interest in technology earlier", "Given your logical mindset", "Building on your interest in software"],
    "MECHANICAL": ["Since you seem to like understanding how things are built", "Given your interest in physical mechanics"],
    "CIVIL": ["Since you mentioned an interest in structures and planning", "Given your focus on infrastructure"],
    "ELECTRICAL": ["Since you seem intrigued by circuits and power", "Building on your interest in electrical systems"],
    "BIOLOGY": ["Since you showed curiosity about living systems", "Given your interest in the natural world"],
    "CHEMISTRY": ["Since you seem interested in how substances interact", "Building on your curiosity about chemical reactions"],
    "PHYSICS": ["Since you showed an interest in the fundamental laws of nature", "Given your analytical approach to physical forces"],
    "MANAGEMENT": ["Since you seem to have strong organizational and leadership interests", "Given your focus on strategy and planning"]
}

class AdaptiveQuestionEngine:
    """
    Manages the adaptive questioning flow for a single counselling session.

    Workflow:
    1. Start with a general opening question.
    2. After each answer, analyze keywords to update career field scores.
    3. Select the next question from the field with the highest emerging score
       (or from an under-explored field to maintain breadth).
    4. After TOTAL_QUESTIONS questions, signal completion.
    """

    def __init__(self):
        """Initialize the engine with empty state."""
        # Career confidence scores (0-100 scale)
        self.field_scores: Dict[str, float] = {
            "CSE": 0.0, "MECHANICAL": 0.0, "CIVIL": 0.0, "ELECTRICAL": 0.0,
            "BIOLOGY": 0.0, "CHEMISTRY": 0.0, "PHYSICS": 0.0, "MANAGEMENT": 0.0,
        }

        # Track which questions have been asked (by question text hash)
        self.asked_questions: set = set()

        # Track all user answers for final prediction
        self.all_answers: List[str] = []

        # Current question number (1-indexed)
        self.question_number: int = 0

        # Track which fields have been explored
        self.explored_fields: Dict[str, int] = {field: 0 for field in self.field_scores}

    def get_welcome_message(self) -> str:
        """Return the chatbot's welcome/introduction message."""
        return (
            "Hello! I'm your AI Career Counsellor. 🎓\n\n"
            "I'm here to help you discover which academic path aligns best with your "
            "natural interests and strengths. I'll ask you a series of thoughtful questions "
            "- there are no right or wrong answers!\n\n"
            "Just be yourself and answer honestly. Let's begin! 🚀"
        )

    def get_first_question(self) -> Tuple[str, List[str]]:
        """Return the first (profiling) question to start the session."""
        self.question_number = 1
        question = "To help me tailor this session perfectly for you, could you share a bit about your current academic journey? For instance, what grade or degree are you currently pursuing?"
        self.asked_questions.add(question)
        suggestions = ["💡 I'm in 10th standard", "💡 I'm in 12th standard", "💡 I'm looking for a bachelor's degree", "💡 Exploring options"]
        return question, suggestions

    def process_answer_and_get_next(self, answer: str) -> Tuple[Optional[str], Optional[List[str]], bool]:
        """
        Process the user's answer and return the next question.

        Args:
            answer: The user's text response.

        Returns:
            Tuple of (next_question_text, suggestions, is_session_complete).
            If session is complete, next_question_text and suggestions are None.
        """
        # Store the answer
        self.all_answers.append(answer)

        # Analyze the answer and update field scores
        self._analyze_answer(answer)

        # Increment question counter
        self.question_number += 1

        # Check if session is complete
        if self.question_number > TOTAL_QUESTIONS:
            return None, None, True

        # Select the next question adaptively
        next_question, suggestions = self._select_next_question()
        return next_question, suggestions, False

    def _analyze_answer(self, answer: str):
        """
        Analyze the user's answers using the actual Machine Learning predictor.
        This ensures the live "Aptitude Score Distribution" perfectly matches
        the final Career Recommendation page.

        Args:
            answer: The user's latest text response.
        """
        # Import inside function to avoid circular dependencies
        from backend.model.predictor import get_predictor
        predictor = get_predictor()

        # Run the ML prediction on ALL answers accumulated so far
        result = predictor.predict(self.all_answers)

        # Update live field scores directly with the ML model's probabilities
        if "all_scores" in result and result["all_scores"]:
            self.field_scores = result["all_scores"]

    def _select_next_question(self) -> Tuple[str, List[str]]:
        """
        Adaptively select the next question based on current scores
        and exploration history.
        """
        # Fixed profiling question for Question 2
        if self.question_number == 2:
            question = "That gives me great context! Thinking about the subjects you've studied recently, which classes did you find yourself naturally looking forward to, and which ones felt like a chore?"
            self.asked_questions.add(question)
            suggestions = ["💡 I loved Science and Math", "💡 Commerce was my favorite", "💡 I enjoyed Arts/Humanities", "💡 I preferred hands-on projects"]
            return question, suggestions

        roll = random.random()

        if self.question_number <= 4:
            # Questions 3 and 4: Mix of general and broad exploration
            target_field = self._get_least_explored_field() if roll > 0.5 else "GENERAL"
        elif roll < 0.6:
            # Dive deeper into the strongest detected interest
            target_field = self._get_top_field()
        elif roll < 0.8:
            # Explore the least covered area
            target_field = self._get_least_explored_field()
        else:
            # Throw in a general question
            target_field = "GENERAL"

        # Get a question from the target field that hasn't been asked yet
        question_data = self._get_unused_question(target_field)

        if question_data is None:
            # Fallback: try any field
            for field in QUESTION_BANK:
                question_data = self._get_unused_question(field)
                if question_data:
                    target_field = field
                    break

        if question_data is None:
            # Ultimate fallback
            return "Tell me more about what motivates you and what kind of work excites you the most?", ["💡 Building things", "💡 Solving problems", "💡 Leading teams", "💡 Helping people"]

        question = question_data["question"]
        suggestions = question_data.get("suggestions", [])
        self.asked_questions.add(question)
        if target_field != "GENERAL":
            self.explored_fields[target_field] = self.explored_fields.get(target_field, 0) + 1

        # Add conversational memory prefix if drilling down into their top field
        if self.question_number >= 5 and random.random() > 0.4:
            top_field = self._get_top_field()
            if target_field == top_field and top_field in MEMORY_PHRASES:
                prefix = random.choice(MEMORY_PHRASES[top_field])
                # Lowercase the first letter of the question for flow
                if question[0].isupper():
                    question = question[0].lower() + question[1:]
                question = f"{prefix}, {question}"

        return question, suggestions

    def _get_top_field(self) -> str:
        """Return the field with the highest confidence score."""
        top = max(self.field_scores, key=self.field_scores.get)
        # If all scores are zero, return a random field
        if self.field_scores[top] == 0:
            return random.choice(list(self.field_scores.keys()))
        return top

    def _get_least_explored_field(self) -> str:
        """Return the career field that has been least explored in questions."""
        return min(self.explored_fields, key=self.explored_fields.get)

    def _get_unused_question(self, field: str) -> Optional[dict]:
        """
        Get a question dict from the specified field that hasn't been asked yet.

        Args:
            field: The career field category.

        Returns:
            The question dict, or None if all questions in this field have been used.
        """
        if field not in QUESTION_BANK:
            return None

        available = [
            q for q in QUESTION_BANK[field]
            if q["question"] not in self.asked_questions
        ]

        if not available:
            return None

        return random.choice(available)

    def get_detected_interests(self) -> Dict[str, float]:
        """
        Return normalized interest scores as percentages.

        Returns:
            Dictionary of {field: percentage} pairs.
        """
        total = sum(self.field_scores.values())
        if total == 0:
            return {field: 12.5 for field in self.field_scores}  # Equal distribution

        return self.field_scores

    def get_all_answers(self) -> List[str]:
        """Return all collected answers for final prediction."""
        return self.all_answers

    def get_progress(self) -> Tuple[int, int]:
        """Return (current_question_number, total_questions)."""
        return min(self.question_number, TOTAL_QUESTIONS), TOTAL_QUESTIONS
