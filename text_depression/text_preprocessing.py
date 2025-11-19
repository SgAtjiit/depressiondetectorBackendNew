import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text for depression detection
    - Convert to lowercase
    - Remove URLs, emails
    - Remove special characters
    - Tokenize
    - Remove stopwords (but keep depression-related negations)
    - Lemmatize
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation but keep sentence structure indicators
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords (but keep important negations)
    important_words = {'not', 'no', 'never', 'nothing', 'nobody', 'none'}
    stop_words = set(stopwords.words('english')) - important_words
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join back to string
    processed_text = ' '.join(tokens)
    
    return processed_text

if __name__ == "__main__":
    # Test preprocessing
    sample_text = """
    I've been feeling really sad lately. I can't sleep at night and I don't enjoy 
    anything anymore. Nothing seems to matter. I feel so hopeless and alone.
    """
    
    print("Original text:")
    print(sample_text)
    print("\nProcessed text:")
    print(preprocess_text(sample_text))