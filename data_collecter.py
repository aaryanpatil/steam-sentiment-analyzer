import requests
import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

# 1. Load Environment Variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# 2. Connect to Database
try:
    client = MongoClient(MONGO_URI)
    db = client["steam_sentiment_db"] # This creates a DB named 'steam_sentiment_db'
    reviews_collection = db["reviews"] # This creates a Collection named 'reviews'
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print(f"❌ Could not connect to MongoDB: {e}")
    exit()

GAME_IDS = {
    "Elden Ring": 1245620,
    "Stardew Valley": 413150,
    "Helldivers 2": 553850
}

def get_steam_reviews(app_id, num_reviews=100):
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    params = {
        'language': 'english',
        'filter': 'recent',
        'num_per_page': 100, 
        'purchase_type': 'all'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200 and response.json()['success'] == 1:
        return response.json()['reviews']
    return []

def save_to_mongo(reviews, game_name):
    if not reviews:
        return
    
    # We prepare a list of "operations" to send to the DB in one batch
    operations = []
    
    for review in reviews:
        # Add the game name to the data
        review['game_name'] = game_name
        
        # Create an update operation:
        # "If you find a review with this recommendationid, update it. If not, insert it."
        op = UpdateOne(
            {"recommendationid": review["recommendationid"]}, # The filter (search condition)
            {"$set": review},                                 # The data to save
            upsert=True                                       # Create if doesn't exist
        )
        operations.append(op)
    
    # Execute all operations at once (much faster than one by one)
    if operations:
        result = reviews_collection.bulk_write(operations)
        print(f"   Saved {len(operations)} reviews for {game_name}. (Inserted: {result.upserted_count}, Modified: {result.modified_count})")

if __name__ == "__main__":
    print("--- Starting Data Collection ---")
    
    for game_name, app_id in GAME_IDS.items():
        print(f"Fetching reviews for {game_name}...")
        reviews = get_steam_reviews(app_id)
        save_to_mongo(reviews, game_name)
        
    print("--- Data Collection Complete ---")