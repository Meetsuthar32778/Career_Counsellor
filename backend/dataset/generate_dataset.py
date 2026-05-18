"""
generate_dataset.py - Synthetic Career Dataset Generator
==========================================================
Generates 1000+ realistic student responses paired with
career field labels for training the ML model.

Usage:
    python -m backend.dataset.generate_dataset
"""

import csv
import random
import os

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "career_dataset.csv")

# ---------------------------------------------------------------------------
# Template pools for each career field
# ---------------------------------------------------------------------------
TEMPLATES = {
    "CSE": {
        "interests": [
            "coding", "programming", "building apps", "web development", "software",
            "artificial intelligence", "machine learning", "data science", "algorithms",
            "hacking", "cybersecurity", "game development", "cloud computing", "databases",
        ],
        "activities": [
            "I love building websites and apps in my free time",
            "I spend hours solving coding challenges on competitive platforms",
            "I enjoy learning new programming languages",
            "I built a chatbot for my school project",
            "I automate repetitive tasks using Python scripts",
            "I find debugging code oddly satisfying",
            "I stay up to date with the latest tech trends",
        ],
        "traits": [
            "logical thinking", "problem solving", "analytical mindset",
            "attention to detail", "systematic approach", "creative with technology",
        ],
    },
    "MECHANICAL": {
        "interests": [
            "engines", "automobiles", "machines", "manufacturing", "CAD design",
            "thermodynamics", "robotics", "fluid mechanics", "turbines", "3D printing",
        ],
        "activities": [
            "I enjoy taking apart machines to understand how they work",
            "I love working with my hands to build things",
            "I spend time in the workshop tinkering with tools",
            "I designed a small drone for a competition",
            "I am fascinated by how car engines convert fuel to motion",
            "I enjoy designing mechanical parts in CAD software",
        ],
        "traits": [
            "hands-on aptitude", "spatial reasoning", "physics intuition",
            "design thinking", "practical problem solving", "curiosity about mechanisms",
        ],
    },
    "CIVIL": {
        "interests": [
            "buildings", "bridges", "roads", "construction", "urban planning",
            "architecture", "structural design", "surveying", "concrete", "dams",
        ],
        "activities": [
            "I notice the design of buildings and infrastructure wherever I go",
            "I dreamed of designing earthquake-resistant structures",
            "I enjoy planning layouts for houses and public spaces",
            "I participated in a bridge-building competition",
            "I find satisfaction in seeing construction projects completed",
            "I research about sustainable and green buildings",
        ],
        "traits": [
            "planning mindset", "environmental awareness", "visual-spatial skills",
            "structural thinking", "community orientation", "detail-oriented",
        ],
    },
    "ELECTRICAL": {
        "interests": [
            "circuits", "electronics", "power systems", "renewable energy", "solar panels",
            "microcontrollers", "Arduino", "voltage", "transformers", "signal processing",
        ],
        "activities": [
            "I enjoy working with circuits and electronic components",
            "I built a small solar-powered device for my project",
            "I am curious about how electricity reaches homes from power plants",
            "I tinker with Arduino and sensors to build smart gadgets",
            "I find electromagnetic theory fascinating",
            "I designed an LED lighting system for energy efficiency",
        ],
        "traits": [
            "analytical precision", "systems thinking", "mathematical aptitude",
            "circuit intuition", "technical curiosity", "interest in energy",
        ],
    },
    "BIOLOGY": {
        "interests": [
            "cells", "DNA", "genetics", "ecology", "human body", "medicine",
            "bacteria", "evolution", "biotechnology", "anatomy", "wildlife",
        ],
        "activities": [
            "I love observing organisms under the microscope",
            "I am fascinated by how the human immune system fights diseases",
            "I enjoy reading about genetic engineering and CRISPR",
            "I volunteered at a wildlife sanctuary during summer",
            "I keep a journal about different plant species I encounter",
            "I dream of finding cures for genetic diseases",
        ],
        "traits": [
            "scientific curiosity", "observation skills", "empathy",
            "research orientation", "detail orientation", "love for nature",
        ],
    },
    "CHEMISTRY": {
        "interests": [
            "chemical reactions", "molecules", "periodic table", "organic chemistry",
            "lab experiments", "polymers", "pharmaceuticals", "titration", "crystals",
        ],
        "activities": [
            "I enjoy conducting experiments and mixing compounds in the lab",
            "I am curious about how medicines interact with our body chemistry",
            "I love predicting products of chemical reactions",
            "I created a pH indicator from natural ingredients",
            "I spend time learning about new materials and alloys",
            "I find the periodic table endlessly interesting",
        ],
        "traits": [
            "experimental mindset", "precision", "analytical thinking",
            "scientific rigor", "curiosity about matter", "lab skills",
        ],
    },
    "PHYSICS": {
        "interests": [
            "gravity", "quantum mechanics", "relativity", "astrophysics",
            "particle physics", "optics", "thermodynamics", "wave theory", "cosmos",
        ],
        "activities": [
            "I think about the forces at play when I watch a ball fly through the air",
            "I am fascinated by black holes and the mysteries of the universe",
            "I enjoy solving complex physics problems with elegant math",
            "I stay up late reading about quantum mechanics",
            "I built a simple telescope to observe celestial bodies",
            "I wonder about the nature of time and space",
        ],
        "traits": [
            "abstract thinking", "mathematical excellence", "theoretical reasoning",
            "curiosity about nature", "problem solving", "love for fundamentals",
        ],
    },
    "MANAGEMENT": {
        "interests": [
            "leadership", "business", "startups", "marketing", "finance",
            "team management", "entrepreneurship", "negotiation", "strategy", "HR",
        ],
        "activities": [
            "I organized the biggest college festival and managed a team of 50 people",
            "I started a small online business and learned about profit margins",
            "I enjoy convincing people and negotiating deals",
            "I created a business plan for a startup idea",
            "I love analyzing market trends and consumer behavior",
            "I mentor juniors and help resolve team conflicts",
        ],
        "traits": [
            "leadership", "communication", "strategic thinking",
            "team collaboration", "decision making", "business acumen",
        ],
    },
}

# ---------------------------------------------------------------------------
# Sentence pattern templates
# ---------------------------------------------------------------------------
PATTERNS = [
    "I am very passionate about {interest}, especially when it comes to {trait}. I also think I have a knack for it.",
    "My favorite projects always involve {interest} and I am fascinated by {interest2}. {activity}.",
    "I find myself naturally drawn to {interest}. {activity}.",
    "{activity}. I believe my strength lies in {trait}.",
    "What excites me most is {interest}. I have always been curious about how things related to {interest2} work.",
    "I want to build a career where I can utilize my skills in {interest}.",
    "In my free time, {activity}. I think {trait} is my biggest strength.",
    "When I think about my future, I see myself working with {interest} and applying {trait}.",
    "{activity}. I am particularly interested in {interest}.",
    "I naturally lean towards {interest}. My friends say I have great {trait}.",
    "I once spent an entire weekend exploring {interest}. {activity}.",
    "If I could change one thing about the world, it would involve {interest}. I approach problems with {trait}.",
]


def generate_sample(label: str) -> str:
    """Generate a single synthetic student response for the given career label."""
    pool = TEMPLATES[label]
    pattern = random.choice(PATTERNS)

    interest = random.choice(pool["interests"])
    interest2 = random.choice([i for i in pool["interests"] if i != interest] or pool["interests"])
    activity = random.choice(pool["activities"])
    trait = random.choice(pool["traits"])

    text = pattern.format(interest=interest, interest2=interest2, activity=activity, trait=trait)
    return text


def generate_dataset(num_samples: int = 1200):
    """Generate the full dataset and save to CSV."""
    labels = list(TEMPLATES.keys())
    samples_per_label = num_samples // len(labels)

    data = []
    for label in labels:
        for _ in range(samples_per_label):
            answer = generate_sample(label)
            data.append({"Answer": answer, "Label": label})

    # Shuffle
    random.shuffle(data)

    # Write CSV
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Answer", "Label"])
        writer.writeheader()
        writer.writerows(data)

    print(f"[Dataset] Generated {len(data)} samples → {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dataset()
