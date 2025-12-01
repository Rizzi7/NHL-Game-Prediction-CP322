import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
model = model_data["model"]
team_stats = model_data["team_stats"]

teams = sorted(team_stats["team"].unique())

# ------------------------------
# Feature Builder
# ------------------------------
def build_matchup(home, away):
    h = team_stats[team_stats["team"] == home].iloc[0]
    a = team_stats[team_stats["team"] == away].iloc[0]

    features = {
        "win_pct_diff": h["win_pct"] - a["win_pct"],
        "xg_pct_diff": h["xg_pct"] - a["xg_pct"],
        "corsi_pct_diff": h["corsi_pct"] - a["corsi_pct"],
    }

    return pd.DataFrame([features])

# ------------------------------
# UI
# ------------------------------
st.title("🏒 NHL Matchup Predictor (Team-Level Model)")

home = st.selectbox("Home Team", teams)
away = st.selectbox("Away Team", [t for t in teams if t != home])

if st.button("Predict"):
    X = build_matchup(home, away)
    prob_home = model.predict_proba(X)[0][1]
    prob_away = 1 - prob_home

    st.subheader("📊 Win Probabilities")
    st.write(f"**{home} Win Probability:** {prob_home*100:.2f}%")
    st.write(f"**{away} Win Probability:** {prob_away*100:.2f}%")

    winner = home if prob_home > 0.5 else away
    st.success(f"🏆 Predicted Winner: **{winner}**")
