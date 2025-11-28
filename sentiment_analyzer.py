import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

# 1. Setup
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["steam_sentiment_db"]
reviews_collection = db["reviews"]

# 2. Load the RoBERTa Model (Hugging Face)
# This downloads the model the first time you run it (approx 400MB)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
print(f"--- Loading Model: {MODEL} ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def get_roberta_sentiment(text):
    # RoBERTa has a limit on text length (usually 512 tokens). 
    # We truncate longer reviews to avoid errors.
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # Run the model
    output = model(**encoded_text)
    
    # The model returns raw scores; we convert them to probabilities (0-1)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
    labels = ['Negative', 'Neutral', 'Positive']
    
    # Get the label with the highest score
    ranking = np.argsort(scores)
    top_label = labels[ranking[-1]]
    
    return {
        "label": top_label,
        "score_negative": float(scores[0]),
        "score_neutral": float(scores[1]),
        "score_positive": float(scores[2])
    }

def analyze_sentiment():
    print("--- Starting Sentiment Analysis (RoBERTa) ---")
    
    # Only find reviews that don't have the 'roberta_label' field yet
    query = {"roberta_label": {"$exists": False}}
    reviews_to_process = list(reviews_collection.find(query))
    
    if not reviews_to_process:
        print("✅ No new reviews to analyze.")
        return

    print(f"Processing {len(reviews_to_process)} new reviews...")
    
    operations = []
    
    for i, review in enumerate(reviews_to_process):
        text = review.get('review', '')
        
        # Skip empty reviews
        if not text.strip():
            continue
            
        try:
            sentiment_result = get_roberta_sentiment(text)
            
            # Prepare the update
            op = UpdateOne(
                {"_id": review["_id"]},
                {"$set": {
                    "roberta_label": sentiment_result['label'],
                    "roberta_pos": sentiment_result['score_positive'],
                    "roberta_neg": sentiment_result['score_negative'],
                    "roberta_neu": sentiment_result['score_neutral']
                }}
            )
            operations.append(op)
            
            # Print progress every 10 reviews (since it's slower)
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1} / {len(reviews_to_process)}")
                
        except Exception as e:
            print(f"⚠️ Error processing review {review['_id']}: {e}")

    # Bulk Update
    if operations:
        result = reviews_collection.bulk_write(operations)
        print(f"✅ Analyzed and updated {result.modified_count} reviews using RoBERTa.")

if __name__ == "__main__":
    analyze_sentiment()