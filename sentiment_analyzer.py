import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException
import numpy as np

# 1. Setup & Connections
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)
    db = client["steam_sentiment_db"]
    reviews_collection = db["reviews"]
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    exit()

# --- MODEL 1: VADER (Best for Slang) ---
vader_analyzer = SentimentIntensityAnalyzer()

# Domain Adaptation
# We inject gaming slang into the lexicon
new_slang = {
    "peak": 4.0,      # Extremely Positive
    "goated": 4.0,    # Greatest of all time
    "mid": -1.5,      # Mediocre/Bad
    "trash": -3.5,    # Terrible
    "garbo": -3.0,    # Garbage
    "broken": -2.0,   # Negative in gaming context
    "buggy": -2.5,
    "unplayable": -4.0
}
vader_analyzer.lexicon.update(new_slang)
print("✅ VADER lexicon updated with gaming slang.")


# --- MODEL 2: RoBERTa (Best for Context) ---
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
print(f"⏳ Loading RoBERTa Model: {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
print("✅ RoBERTa Loaded.")

def get_roberta_sentiment(text):
    try:
        encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        labels = ['Negative', 'Neutral', 'Positive']
        ranking = np.argsort(scores)
        top_label = labels[ranking[-1]]
        
        return {
            "label": top_label,
            "confidence": float(scores[ranking[-1]])
        }
    except Exception as e:
        print(f"RoBERTa Error: {e}")
        return {"label": "Neutral", "confidence": 0.0}

def is_actually_english(text):
    """
    Steam says it's English, but text may still have different language.
    This checks if the text is actually English.
    """
    try:
        # Check if text is long enough to detect
        if len(text) < 3: 
            return True # Assume short slang like "GG" is fine
        return detect(text) == 'en'
    except LangDetectException:
        return False

def analyze_sentiment():
    print("--- Starting Hybrid Sentiment Analysis ---")
    
    # Reset analysis for testing? Uncomment next line to re-analyze EVERYTHING
    reviews_collection.update_many({}, {"$unset": {"roberta_label": "", "vader_score": ""}})

    # Find reviews that don't have a 'roberta_label' yet
    query = {"roberta_label": {"$exists": False}}
    reviews_to_process = list(reviews_collection.find(query))
    
    if not reviews_to_process:
        print("✅ No new reviews to analyze.")
        return

    print(f"Processing {len(reviews_to_process)} reviews...")
    
    operations = []
    
    for i, review in enumerate(reviews_to_process):
        text = review.get('review', '')
        
        # 1. Cleanliness Check
        if not text.strip() or not is_actually_english(text):
            # Mark as skipped so we don't process it again
            op = UpdateOne({"_id": review["_id"]}, {"$set": {"skipped": True, "reason": "Not English/Empty"}})
            operations.append(op)
            continue

        # 2. Run VADER (Fast, Good for "PEAK")
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # 3. Run RoBERTa (Slow, Good for Context)
        roberta_result = get_roberta_sentiment(text)
        
        # 4. Save Both
        op = UpdateOne(
            {"_id": review["_id"]},
            {"$set": {
                "vader_score": vader_compound, # -1 to 1
                "roberta_label": roberta_result['label'], # Neg/Neu/Pos
                "roberta_confidence": roberta_result['confidence'],
                "analyzed_at": "hybrid_v2" # Version control your data!
            }}
        )
        operations.append(op)
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1} / {len(reviews_to_process)}")

    if operations:
        result = reviews_collection.bulk_write(operations)
        print(f"✅ Modified {result.modified_count} documents.")

if __name__ == "__main__":
    analyze_sentiment()