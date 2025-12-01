import streamlit as st
import pickle
import pandas as pd
from config import FEATURE_COLS

# Load model and data
@st.cache_resource
def load_model():
    with open('nhl_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('nhl_data.csv')

# Load everything
model_data = load_model()
model = model_data['model']
FEATURE_COLS = model_data['feature_cols']

df = load_data()

# Prediction function
def predict_game(team_a, team_b, is_team_a_home=True):
    # Get most recent game for each team
    team_a_data = df[df['playerTeam'] == team_a].sort_values('gameDate', ascending=False).iloc[0]
    team_b_data = df[df['playerTeam'] == team_b].sort_values('gameDate', ascending=False).iloc[0]
    
    # Build feature vector: team_a stats - team_b stats
    features = {}
    for col in FEATURE_COLS:
        if 'home' in col.lower():  # Handle home/away indicator
            features[col] = 1 if is_team_a_home else 0
        else:
            features[col] = team_a_data[col] - team_b_data[col]
    
    # Convert to DataFrame and predict
    features_df = pd.DataFrame([features])[FEATURE_COLS]
    proba = model.predict_proba(features_df)[0]
    
    return {
        team_a: proba[1],
        team_b: proba[0]
    }

# UI
st.title('NHL Game Predictor')
st.write('Predict the outcome of an NHL matchup based on recent team performance')

teams = sorted(df['playerTeam'].unique())

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox('Home Team', teams, key='home')
    
with col2:
    team_b = st.selectbox('Away Team', teams, key='away')

if st.button('Predict Winner'):
    if team_a == team_b:
        st.error('Please select two different teams!')
    else:
        result = predict_game(team_a, team_b, is_team_a_home=True)
        
        st.subheader('Prediction Results')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label=f"{team_a} (Home)",
                value=f"{result[team_a]:.1%}"
            )
        
        with col2:
            st.metric(
                label=f"{team_b} (Away)",
                value=f"{result[team_b]:.1%}"
            )
        
        winner = team_a if result[team_a] > result[team_b] else team_b
        confidence = max(result[team_a], result[team_b])
        
        st.success(f"**Predicted Winner: {winner}** (Confidence: {confidence:.1%})")