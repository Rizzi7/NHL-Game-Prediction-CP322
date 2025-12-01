import streamlit as st
import pickle
import pandas as pd

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

def predict_game(home_team, away_team):
    matchup = (
        df[(df['home_team'] == home_team) & (df['away_team'] == away_team)]
        .sort_values('gameDate_home', ascending=False)
    )

    if matchup.empty:
        raise ValueError(f"No historical games found for {home_team} (home) vs {away_team} (away).")

    row = matchup.iloc[0]

    # Ensure numeric dtypes for XGBoost
    features_series = row[FEATURE_COLS].astype(float)
    features_df = features_series.to_frame().T  # shape (1, n_features)

    proba = model.predict_proba(features_df)[0]

    classes = list(model.classes_)
    home_win_idx = classes.index(1)
    p_home = proba[home_win_idx]
    p_away = 1.0 - p_home

    return {
        home_team: p_home,
        away_team: p_away
    }

# UI
st.title('NHL Game Predictor')
st.write('Predict the outcome of an NHL matchup based on recent team performance')

teams = sorted(df['home_team'].unique())

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox('Home Team', teams, key='home')
    
with col2:
    team_b = st.selectbox('Away Team', teams, key='away')

if st.button('Predict Winner'):
    if team_a == team_b:
        st.error('Please select two different teams!')
    else:
        result = predict_game(team_a, team_b)
        
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