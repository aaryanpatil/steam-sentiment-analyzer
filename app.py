import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import plotly.express as px  # <--- The new library

# 1. Page Configuration
st.set_page_config(
    page_title="Steam Sentiment Analyzer",
    page_icon="🎮",
    layout="wide"
)

# Custom Color Palette for Sentiment
COLOR_MAP = {
    "Positive": "#00CC96",  # Green
    "Neutral": "#636EFA",   # Blue
    "Negative": "#EF553B"   # Red
}

# 2. Load Data (Cached)
@st.cache_data
def load_data():
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client["steam_sentiment_db"]
    collection = db["reviews"]
    
    # Fetch data with roberta_label
    data = list(collection.find({"roberta_label": {"$exists": True}}))
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df['_id'] = df['_id'].astype(str)
    
    # Calculate Review Length for analysis
    df['review_length'] = df['review'].str.len()
    
    return df

# 3. Sidebar
st.sidebar.header("🔍 Filter Options")
df = load_data()

if df.empty:
    st.warning("No analyzed data found in MongoDB! Run sentiment_analyzer.py first.")
    st.stop()

unique_games = df['game_name'].unique()
selected_game = st.sidebar.selectbox("Select a Game", unique_games)

# Filter Data
game_df = df[df['game_name'] == selected_game]

# 4. Main Dashboard layout
st.title(f"🎮 Sentiment Analysis: {selected_game}")
st.markdown("### Powered by RoBERTa (Transformer Model)")

# --- KPI Row ---
col1, col2, col3 = st.columns(3)

total_reviews = len(game_df)
positive_count = len(game_df[game_df['roberta_label'] == 'Positive'])
positive_percent = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0
avg_len = game_df['review_length'].mean()

with col1:
    st.metric("Total Reviews", total_reviews)
with col2:
    st.metric("Positive Sentiment", f"{positive_percent:.1f}%")
with col3:
    st.metric("Avg Review Length", f"{int(avg_len)} chars")

st.divider()

# --- Interactive Charts Row 1 ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
    # Donut Chart
    fig_pie = px.pie(
        game_df, 
        names='roberta_label', 
        color='roberta_label',
        color_discrete_map=COLOR_MAP,
        hole=0.4 # Makes it a donut
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Sentiment Confidence")
    # Box Plot to show how "sure" the model was
    # This helps find "weak" predictions
    fig_box = px.box(
        game_df, 
        x='roberta_label', 
        y='roberta_pos', # Using Positive score as a proxy for confidence
        color='roberta_label',
        color_discrete_map=COLOR_MAP,
        points="all", # Show individual dots
        title="Model Confidence (Positive Probability)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- Interactive Charts Row 2 ---
st.subheader("Does Length Matter?")
# Histogram to see if long reviews are different
fig_hist = px.histogram(
    game_df, 
    x="review_length", 
    color="roberta_label",
    color_discrete_map=COLOR_MAP,
    nbins=30,
    title="Distribution of Review Lengths by Sentiment",
    labels={"review_length": "Character Count"}
)
st.plotly_chart(fig_hist, use_container_width=True)

# --- Raw Data Table ---
st.divider()
st.subheader("📝 Recent Reviews Explorer")
st.dataframe(
    game_df[['review', 'roberta_label', 'review_length']],
    use_container_width=True,
    hide_index=True
)