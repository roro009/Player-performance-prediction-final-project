#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the dataset
game_details = pd.read_csv('games_details.csv')

# Check for missing values
for col in game_details:
    print(col, game_details[col].isnull().sum()/game_details.shape[0], "%")

# Handle missing values
# Example: Impute missing numerical values with mean
game_details.fillna(game_details.mean(), inplace=True)

# Explore the distribution of numerical features
print("Summary Statistics:\n", game_details.describe())


# In[9]:


game_details

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Function to convert 'MIN' to float
def convert_min_to_float(min_str):
    if pd.isna(min_str):
        return None
    try:
        parts = min_str.split(':')
        return int(parts[0]) + int(parts[1]) / 60
    except:
        return None

# Streamlit app title
st.title('Basketball Player Performance Prediction')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Read data
    games_data = pd.read_csv(uploaded_file)

    # Data preprocessing
    games_data['MIN'] = games_data['MIN'].apply(convert_min_to_float)
    # Add additional preprocessing as needed

    # Feature Engineering
    n_games = 5
    player_avg_stats = games_data.groupby(['PLAYER_ID', 'GAME_ID'])[['MIN', 'FGM', 'FGA', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS']].rolling(n_games, min_periods=1).mean().reset_index()
    games_data_merged = games_data.merge(player_avg_stats, on=['PLAYER_ID', 'GAME_ID'], suffixes=('', '_avg'))
    games_data_merged = games_data_merged.dropna(subset=['PTS_avg'])

    # Selecting features and target
    features = games_data_merged[['MIN_avg', 'FGM_avg', 'FGA_avg', 'REB_avg', 'AST_avg', 'STL_avg', 'BLK_avg', 'TO_avg', 'PF_avg']]
    target = games_data_merged['PTS_avg']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Making predictions and evaluating the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Displaying the results
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")





