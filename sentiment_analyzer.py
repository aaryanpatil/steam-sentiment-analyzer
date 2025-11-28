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
new_slang = {
    "peak": 4.0, "goated": 4.0, "mid": -1.5, "trash": -3.5, 
    "garbo": -3.0, "broken": -2.0, "buggy": -2.5, "unplayable": -4.0, "goty": 4.0
}
vader_analyzer.lexicon.update(new_slang)
print("✅ VADER lexicon updated.")

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
    Revised to handle short gaming reviews like 'GG', 'Nice', '10/10'.
    """
    # 1. Short Review Protection (< 25 chars)
    # Captures: "Good Game", "Nice", "Peak", "Trash", "10/10"
    if len(text) <= 25: 
        return True 

    try:
        # 2. Run detection on longer text
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return False

def analyze_sentiment():
    print("--- Starting Hybrid Sentiment Analysis ---")
    
    # --- RESET PREVIOUSLY SKIPPED REVIEWS ---
    # This is crucial. We must look for reviews that were marked "skipped" 
    # but might actually be valid short English reviews.
    print("Resetting previously 'Skipped' reviews to try again...")
    reviews_collection.update_many(
        {"skipped": True}, 
        {"$unset": {"skipped": "", "reason": ""}}
    )

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
            op = UpdateOne({"_id": review["_id"]}, {"$set": {"skipped": True, "reason": "Not English/Empty"}})
            operations.append(op)
            continue

        # 2. Run VADER
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # 3. Run RoBERTa
        roberta_result = get_roberta_sentiment(text)
        
        # 4. Save Both
        op = UpdateOne(
            {"_id": review["_id"]},
            {"$set": {
                "vader_score": vader_compound, 
                "roberta_label": roberta_result['label'], 
                "roberta_confidence": roberta_result['confidence'],
                "analyzed_at": "hybrid_v3",
                # Remove skipped flag if it existed previously
                "skipped": False 
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