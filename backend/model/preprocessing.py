"""
preprocessing.py - Text Preprocessing Utilities
==================================================
Functions for cleaning and normalizing student responses
before embedding generation.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean and normalize a text string.
    
    Steps:
    1. Convert to lowercase
    2. Remove special characters (keep alphanumeric and spaces)
    3. Remove extra whitespace
    4. Strip leading/trailing spaces
    
    Args:
        text: Raw text input from student.
    
    Returns:
        Cleaned text string.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def validate_answer(text: str) -> bool:
    """
    Validate that a student's answer is meaningful.
    
    Args:
        text: The student's answer.
    
    Returns:
        True if the answer has at least 3 words, False otherwise.
    """
    words = text.strip().split()
    return len(words) >= 3
