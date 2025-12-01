import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load model bundle
# -----------------------------
@st.cache_resource
def load_bundle():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()

model          = bundle["model"]
feature_names  = bundle["feature_names"]   # list like ['xGoalsPercentage_diff', 'corsiPercentage_diff']
team_summary   = bundle["team_summary"]    # season-to-date team averages
base_cols      = bundle["base_cols"]       # ['xGoalsPercentage', 'corsiPercentage']

# -----------------------------
# App UI
# -----------------------------
st.title("🏒 NHL Win Predictor — XGBoost Model")

teams = sorted(team_summary["team"].unique())

home = st.selectbox("Home Team", teams)
away = st.selectbox("Away Team", [t for t in teams if t != home])

# -----------------------------
# Build feature vector (home-away diffs)
# -----------------------------
def build_features(home_team, away_team):
    h = team_summary[team_summary["team"] == home_team].iloc[0]
    a = team_summary[team_summary["team"] == away_team].iloc[0]

    feature_dict = {}

    # compute diffs for all base columns: home minus away
    for col in base_cols:
        diff_val = float(h[col]) - float(a[col])
        feature_dict[f"{col}_diff"] = diff_val

    # reorder for model
    X = pd.DataFrame([[feature_dict[col] for col in feature_names]],
                     columns=feature_names)

    return X

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    X = build_features(home, away)
    prob_home = float(model.predict_proba(X)[0][1])
    prob_away = 1 - prob_home

    st.subheader("📊 Win Probabilities")
    st.write(f"**{home} (HOME)**: {prob_home*100:.2f}%")
    st.write(f"**{away} (AWAY)**: {prob_away*100:.2f}%")

    if prob_home > 0.5:
        st.success(f"🏆 Predicted Winner: **{home}**")
    else:
        st.success(f"🏆 Predicted Winner: **{away}**")

    with st.expander("Show model input features"):
        st.dataframe(X)
