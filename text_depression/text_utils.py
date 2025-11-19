import os
import pandas as pd
from text_config import DEPRESSION_KEYWORDS

def count_depression_keywords(text):
    """
    Count depression-related keywords in text
    """
    text_lower = text.lower()
    count = sum(1 for keyword in DEPRESSION_KEYWORDS if keyword in text_lower)
    return count

def get_depression_risk_level(probability):
    """
    Determine risk level based on probability
    """
    if probability >= 0.75:
        return "High Risk"
    elif probability >= 0.50:
        return "Moderate Risk"
    elif probability >= 0.30:
        return "Low Risk"
    else:
        return "Minimal Risk"

def analyze_text_sentiment(text):
    """
    Simple sentiment analysis
    """
    negative_words = ['sad', 'unhappy', 'depressed', 'miserable', 'hopeless', 
                     'worthless', 'lonely', 'empty', 'numb', 'tired']
    positive_words = ['happy', 'joy', 'excited', 'great', 'wonderful', 
                     'good', 'better', 'hopeful', 'grateful', 'love']
    
    text_lower = text.lower()
    
    neg_count = sum(1 for word in negative_words if word in text_lower)
    pos_count = sum(1 for word in positive_words if word in text_lower)
    
    if neg_count > pos_count:
        return "Negative"
    elif pos_count > neg_count:
        return "Positive"
    else:
        return "Neutral"

if __name__ == "__main__":
    # Test utility functions
    sample_text = "I feel so sad and hopeless. Nothing makes me happy anymore."
    
    print(f"Text: {sample_text}")
    print(f"Depression keywords: {count_depression_keywords(sample_text)}")
    print(f"Sentiment: {analyze_text_sentiment(sample_text)}")
    print(f"Risk level (0.8 prob): {get_depression_risk_level(0.8)}")