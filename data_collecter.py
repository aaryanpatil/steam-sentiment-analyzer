import requests
import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from urllib.parse import quote # Needed for the cursor encoding

# 1. Setup
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)
    db = client["steam_sentiment_db"]
    reviews_collection = db["reviews"]
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print(f"❌ Connection Error: {e}")
    exit()

# Add as many games as you want here
GAME_IDS = {
    "Elden Ring": 1245620,
    "Stardew Valley": 413150,
    "Helldivers 2": 553850,
    "Cyberpunk 2077": 1091500, 
    "Baldur's Gate 3": 1086940,
    "Apex Legends": 1172470,
    "Counter-Strike 2": 730,
    "Dota 2": 570,
    "Red Dead Redemption 2": 1174180
}

def get_reviews_with_pagination(app_id, target_count=500):
    """
    Fetches reviews using a cursor to get more than 100.
    """
    reviews_collected = []
    cursor = "*" # The API uses '*' to signify "start from the beginning"
    
    print(f"   Targeting {target_count} reviews for AppID {app_id}...")

    while len(reviews_collected) < target_count:
        url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
        
        params = {
            'language': 'english',
            'filter': 'recent',
            'num_per_page': 100, # Max allowed per request
            'purchase_type': 'all',
            'cursor': cursor # This tells Steam where we left off
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['success'] == 1 and len(data['reviews']) > 0:
                # Add the new batch to our list
                reviews_collected.extend(data['reviews'])
                
                # Update the cursor for the next loop
                cursor = data['cursor']
                
                print(f"   ...Fetched {len(reviews_collected)} / {target_count}")
                
                # Be polite to the API (Anti-ban protection)
                time.sleep(1) 
            else:
                # No more reviews available
                print("   ⚠️ No more reviews available from Steam.")
                break
                
        except Exception as e:
            print(f"   ❌ Network error: {e}")
            break
            
    # Trim to exactly the target count if we went over
    return reviews_collected[:target_count]

def save_to_mongo(reviews, game_name):
    if not reviews:
        return
    
    operations = []
    for review in reviews:
        review['game_name'] = game_name
        
        # Upsert: Prevent duplicates, preserve existing sentiment analysis
        op = UpdateOne(
            {"recommendationid": review["recommendationid"]},
            {"$set": review},
            upsert=True
        )
        operations.append(op)
    
    if operations:
        result = reviews_collection.bulk_write(operations)
        # Note: 'Modified' means existing reviews updated. 'Upserted' means new ones added.
        print(f"   💾 Database: {result.upserted_count} new, {result.modified_count} updated.")

if __name__ == "__main__":
    print("--- Starting Large Batch Data Collection ---")
    
    for game_name, app_id in GAME_IDS.items():
        print(f"\n🎮 Processing: {game_name}")
        
        # Fetch 500 reviews instead of 100
        reviews = get_reviews_with_pagination(app_id, target_count=500)
        
        save_to_mongo(reviews, game_name)
        
    print("\n--- Collection Complete ---")