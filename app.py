import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import plotly.express as px

# 1. Page Config & Setup
st.set_page_config(page_title="Steam Review Analyzer", page_icon="🎮", layout="wide")

COLOR_MAP = {
    "Positive": "#00CC96", 
    "Neutral": "#636EFA",   
    "Negative": "#EF553B",
    "Skipped": "#7f7f7f"  # Grey for unanalyzed/non-English data
}

# 2. Load Data (Updated to include Skipped/Non-English reviews)
@st.cache_data
def load_data():
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    
    if not MONGO_URI:
        return pd.DataFrame()
        
    client = MongoClient(MONGO_URI)
    db = client["steam_sentiment_db"]
    collection = db["reviews"]
    
    # FETCH EVERYTHING (Remove the filter)
    # We want to see the raw 500 count, even if not analyzed
    data = list(collection.find({})) 
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    
    # --- TYPE FIXING ---
    df['_id'] = df['_id'].astype(str)
    
    numeric_cols = [
        'weighted_vote_score', 'vader_score', 
        'roberta_confidence', 'roberta_pos', 'roberta_neg', 'roberta_neu'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # --- HANDLING SKIPPED DATA ---
    # If roberta_label is missing, it means the analyzer skipped it (Non-English/Empty)
    if 'roberta_label' not in df.columns:
        df['roberta_label'] = 'Skipped' # Handle case where NO analysis exists yet
    else:
        df['roberta_label'] = df['roberta_label'].fillna('Skipped')

    df['review_length'] = df['review'].str.len()
    
    # Create VADER Label with a safety check for Skipped rows
    def get_vader_label(row):
        if row['roberta_label'] == 'Skipped':
            return 'Skipped'
        score = row.get('vader_score', 0.0)
        if score >= 0.05: return "Positive"
        elif score <= -0.05: return "Negative"
        else: return "Neutral"
        
    df['vader_label'] = df.apply(get_vader_label, axis=1)
    
    # Comparison Logic (Skipped rows don't count as agreement)
    df['models_agree'] = (df['roberta_label'] == df['vader_label']) & (df['roberta_label'] != 'Skipped')
    
    return df

    # 3. Calculate lengths
    df['review_length'] = df['review'].str.len()
    
    # 4. Create Comparison Logic
    def get_vader_label(score):
        if score >= 0.05: return "Positive"
        elif score <= -0.05: return "Negative"
        else: return "Neutral"
        
    df['vader_label'] = df['vader_score'].apply(get_vader_label)
    df['models_agree'] = df['roberta_label'] == df['vader_label']
    
    return df

# 3. Sidebar
st.sidebar.title("🎮 Steam Review Analyzer Dashboard")
df = load_data()

if df.empty:
    st.warning("No hybrid analyzed data found! Run the new sentiment_analyzer.py first.")
    st.stop()

# Game Selector
games = sorted(df['game_name'].unique())
selected_game = st.sidebar.selectbox("Select Game", games)
game_df = df[df['game_name'] == selected_game]

# 4. Main UI
st.title(f"Sentiment Analysis: {selected_game}")
st.caption("Comparing Dictionary-based (VADER) vs Transformer-based (RoBERTa) models")

# --- Tabs for Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "⚔️ Model Comparison", "📝 Raw Data"])

with tab1:
    # KPI Metrics
    total = len(game_df)
    pos_roberta = len(game_df[game_df['roberta_label'] == 'Positive'])
    pos_vader = len(game_df[game_df['vader_label'] == 'Positive'])
    
    # Safe division
    pct_roberta = (pos_roberta / total) * 100 if total > 0 else 0
    pct_vader = (pos_vader / total) * 100 if total > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", total)
    c2.metric("RoBERTa Positive %", f"{pct_roberta:.1f}%")
    c3.metric("VADER Positive %", f"{pct_vader:.1f}%", delta=f"{pct_vader - pct_roberta:.1f}% vs RoBERTa")
    
    st.divider()
    
   # Dual Donut Charts
    c_chart1, c_chart2 = st.columns(2)
    
    # Define the strict order we want
    sentiment_order = ["Positive", "Neutral", "Negative", "Skipped"]
    
    with c_chart1:
        st.subheader("RoBERTa (Context Aware)")
        fig_rob = px.pie(
            game_df, 
            names='roberta_label', 
            color='roberta_label', 
            color_discrete_map=COLOR_MAP, 
            hole=0.5,
            # THIS FORCE-SORTS THE SLICES
            category_orders={'roberta_label': sentiment_order} 
        )

        # --- CUSTOM HOVER TEXT ---
        fig_rob.update_traces(
            textinfo='percent',             # Show % on the chart itself
            hoverinfo='label+percent+name', # Standard info
            hovertemplate='%{label}: <br><b>%{value} reviews</b><br>(%{percent})' # Custom HTML
        )
        
        st.plotly_chart(fig_rob, width='stretch')
        
    with c_chart2:
        st.subheader("VADER (Slang Aware)")
        fig_vad = px.pie(
            game_df, 
            names='vader_label', 
            color='vader_label', 
            color_discrete_map=COLOR_MAP, 
            hole=0.5,
            # THIS FORCE-SORTS THE SLICES
            category_orders={'vader_label': sentiment_order}
        )

        # --- CUSTOM HOVER TEXT ---
        fig_vad.update_traces(
            textinfo='percent',
            hovertemplate='%{label}: <br><b>%{value} reviews</b><br>(%{percent})'
        )

        st.plotly_chart(fig_vad, width="stretch")

with tab2:
    st.header("Where do the models disagree?")
    
    # Calculate Disagreement
    disagreement_df = game_df[~game_df['models_agree']]
    disagreement_rate = (len(disagreement_df) / total) * 100 if total > 0 else 0
    
    st.info(f"The models disagree on **{disagreement_rate:.1f}%** of reviews.")
    
    # Scatter Plot
    fig_scatter = px.scatter(
        game_df,
        x="vader_score",
        y="roberta_confidence",
        color="roberta_label",
        color_discrete_map=COLOR_MAP,
        hover_data=["review"],
        title="Model Correlation: VADER Score vs RoBERTa Confidence"
    )
    fig_scatter.add_vline(x=0.05, line_dash="dash", line_color="gray")
    fig_scatter.add_vline(x=-0.05, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_scatter, width='stretch')
    
    st.subheader("🔍 Review Inspector")
    filter_type = st.radio("Show reviews where:", ["Models Agree", "Models Disagree", "Contains Slang ('Peak', 'Mid')"])
    
    if filter_type == "Models Agree":
        show_df = game_df[game_df['models_agree']]
    elif filter_type == "Models Disagree":
        show_df = disagreement_df
    else:
        # Safe slang filter
        show_df = game_df[game_df['review'].str.lower().str.contains("peak|mid|trash|goat", regex=True, na=False)]
    
    # Updated: Replaced use_container_width=True with width="stretch" (or remove to default)
    # Streamlit recommends specific keyword args for Dataframes now
    st.dataframe(
        show_df[['review', 'roberta_label', 'vader_label', 'vader_score']],
        hide_index=True
    )

with tab3:
    st.subheader("Full Dataset")
    # Updated: Removed use_container_width=True
    st.dataframe(game_df)