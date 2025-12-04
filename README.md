# NHL Game Prediction (CP322 Machine Learning Project)

This project predicts NHL game outcomes (win/loss) using team-level performance statistics from the [MoneyPuck](https://moneypuck.com/data.htm) dataset.

---

Abstract - This project looks at the challenge of predicting the outcomes of NHL games using machine learning techniques. We explore multiple models which include logistic regression and random forest to give us a prediction on game winners based on historical game data. XGBoost is implemented as a novel approach. This historical game data was taken from several recent NHL seasons, which featured information such as goals and shots. After training and evaluating both these models we can see that XGBoost consistently outperforms the logistic regression baseline by offering a higher accuracy and more reliable predictions on unseen data. This can lead us to believe non-linear methods may be better at capturing complex patterns which are present in hockey games. Overall, our findings suggest that shot-quality metrics (Corsi%, xG%) and special-teams performance are the most influential factors in predicting outcomes

## Overview
The goal of this project is to build and evaluate machine learning models that can predict whether a team will win or lose an NHL game based on features such as:
- Shots on goal
- Expected goals (xG)
- Penalties
- Corsi%
- Other advanced performance metrics

The project follows a complete machine learning pipeline:
1. **Data Preparation & Exploration**
2. **Feature Engineering**
3. **Model Training (Logistic Regression, Random Forest, XGBoost)**
4. **Evaluation (Accuracy, ROC-AUC, SHAP Analysis)**
5. **Visualization & Interpretation**

---

## 📊 Dataset
The dataset is available from [MoneyPuck.com](https://moneypuck.com/data.htm).  
Please download `all_teams.csv` and place it in the following path: NHL-Game-Prediction-CP322/data/raw 

Use "streamlit run app.py" to run

