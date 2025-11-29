import streamlit as st
import pickle
import pandas as pd

# ===== LOAD MODEL AND DATA =====
@st.cache_resource
def load_model():
    with open('nhl_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    # YOU NEED TO: Save your processed df to a CSV first
    # In your notebook: df.to_csv('nhl_data.csv', index=False)
    return pd.read_csv('nhl_data.csv')

model = load_model()
df = load_data()

# ===== HELPER FUNCTION: Get team's most recent stats =====
def get_team_recent_stats(team_name, df):
    """
    Gets the latest rolling stats for a team by finding their most recent game
    """
    # Filter for this team's games, sort by date, take the most recent
    team_data = df[df['playerTeam'] == team_name].sort_values('gameDate', ascending=False).iloc[0]
    
    # Return a dictionary of their current rolling stats
    # IMPORTANT: These column names must match YOUR dataframe columns
    return {
        'xGoalsFor_last5': team_data['xGoalsFor_last5'],
        'xGoalsAgainst_last5': team_data['xGoalsAgainst_last5'],
        'xGoalsFor_slope5': team_data['xGoalsFor_slope5'],
        'xGoalsAgainst_slope5': team_data['xGoalsAgainst_slope5'],
        'shotsOnGoalFor_slope5': team_data['shotsOnGoalFor_slope5'],
        'shotsOnGoalAgainst_slope5': team_data['shotsOnGoalAgainst_slope5'],
        # ADD ALL OTHER FEATURES YOU USED IN TRAINING HERE
    }

# ===== PREDICTION FUNCTION =====
def predict_game(team_a, team_b, df, model, is_team_a_home=True):
    """
    Predicts win probability for team_a vs team_b
    
    What this does:
    1. Gets recent stats for both teams
    2. Calculates differences (team_a - team_b) for each stat
    3. Feeds those differences to the model
    4. Returns win probabilities
    """
    
    # Step 1: Get current stats for both teams
    team_a_stats = get_team_recent_stats(team_a, df)
    team_b_stats = get_team_recent_stats(team_b, df)
    
    # Step 2: Calculate differences (this matches how you trained)
    # The model was trained on "home_stat - away_stat"
    features = [[
        team_a_stats['xGoalsFor_last5'] - team_b_stats['xGoalsFor_last5'],
        team_a_stats['xGoalsAgainst_last5'] - team_b_stats['xGoalsAgainst_last5'],
        team_a_stats['xGoalsFor_slope5'] - team_b_stats['xGoalsFor_slope5'],
        team_a_stats['xGoalsAgainst_slope5'] - team_b_stats['xGoalsAgainst_slope5'],
        team_a_stats['shotsOnGoalFor_slope5'] - team_b_stats['shotsOnGoalFor_slope5'],
        team_a_stats['shotsOnGoalAgainst_slope5'] - team_b_stats['shotsOnGoalAgainst_slope5'],
        # ADD ALL OTHER FEATURE DIFFERENCES HERE
        # MUST MATCH THE ORDER YOU USED IN feature_cols DURING TRAINING
        1 if is_team_a_home else 0  # Home team indicator
    ]]
    
    # Step 3: Get prediction probabilities
    # predict_proba returns [prob_loss, prob_win]
    proba = model.predict_proba(features)[0]
    
    # Step 4: Return as dictionary
    return {
        team_a: proba[1],  # Probability team_a wins
        team_b: proba[0]   # Probability team_b wins  
    }

# ===== UI =====
st.title('🏒 NHL Game Predictor')
st.write('Predict the outcome of an NHL matchup based on recent team performance')

# Get unique team names from your data
teams = sorted(df['playerTeam'].unique())

# Two columns for team selection
col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox('Home Team', teams, key='home')
    
with col2:
    team_b = st.selectbox('Away Team', teams, key='away')

# Predict button
if st.button('Predict Winner'):
    if team_a == team_b:
        st.error('Please select two different teams!')
    else:
        # Get prediction
        result = predict_game(team_a, team_b, df, model, is_team_a_home=True)
        
        st.subheader('📊 Prediction Results')
        
        # Display probabilities side by side
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
        
        # Show predicted winner
        winner = team_a if result[team_a] > result[team_b] else team_b
        confidence = max(result[team_a], result[team_b])
        
        st.success(f"**Predicted Winner: {winner}** (Confidence: {confidence:.1%})")