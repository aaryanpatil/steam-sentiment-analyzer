import requests
import json

# 1. Define the Games we want to track
GAME_IDS = {
    "Elden Ring": 1245620,
    "Stardew Valley": 413150,
    "Helldivers 2": 553850
}

def get_steam_reviews(app_id, num_reviews=100):
    """
    Fetches the most recent reviews for a specific game.
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    
    # Parameters for the API call
    params = {
        'language': 'english',   # We only want English for NLP
        'filter': 'recent',      # Get the latest reviews
        'num_per_page': 100,     # Max allowed per request
        'purchase_type': 'all'   # Steam purchase or Key activation
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Check if the query was successful
        if data['success'] == 1:
            reviews = data['reviews']
            print(f"✅ Successfully fetched {len(reviews)} reviews for AppID: {app_id}")
            return reviews
        else:
            print(f"❌ API returned success=0 for AppID: {app_id}")
            return []
    else:
        print(f"❌ Failed to connect. Status Code: {response.status_code}")
        return []

# 2. Main Execution Block
if __name__ == "__main__":
    print("--- Starting Data Collection ---")
    
    # Temporary storage to see if it works
    all_reviews = []
    
    for game_name, app_id in GAME_IDS.items():
        print(f"Fetching reviews for {game_name}...")
        game_reviews = get_steam_reviews(app_id)
        
        # Let's tag each review with the game name (useful for the DB later)
        for review in game_reviews:
            review['game_name'] = game_name
            
        all_reviews.extend(game_reviews)

    # 3. Save to a local file for inspection (sanity check)
    with open('temp_reviews.json', 'w', encoding='utf-8') as f:
        json.dump(all_reviews, f, indent=4)
        
    print(f"--- Done! Saved {len(all_reviews)} reviews to temp_reviews.json ---")