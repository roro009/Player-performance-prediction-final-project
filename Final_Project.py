#!/usr/bin/env python
# coding: utf-8

# # MVP

# ## About Dataset
# Context
# 
# This dataset was collected to work on NBA games data. I used the nba stats website to create this dataset.
# 
# You can find more details about data collection in my GitHub repo here : nba predictor repo.
# 
# If you want more informations about this api endpoint feel free to go on the nba_api GitHub repo that documentate each endpoint : link here
# 
# Content
# 
# You can find 5 datasets :
# 
# games.csv : all games from 2004 season to last update with the date, teams and some details like number of points, etc.
# 
# games_details.csv : details of games dataset, all statistics of players for a given game
# 
# players.csv : players details (name)
# 
# ranking.csv : ranking of NBA given a day (split into west and east on CONFERENCE column
# 
# teams.csv : all teams of NBA
# 
# Acknowledgements:
# 
# I would like to thanks nba stats website which allows all NBA data freely open to everyone and with a great api endpoint.

# ## Problem Statement:
# 
# The goal of this project is to create a predictive model that, using a variety of game-related characteristics, can reliably predict how many points a basketball player will score during a game. Performance statistics, player and team details, and other data are included in the game_details dataset. The objective is to develop a model that predicts a player's total points scored in a game based on input features such as field goals made, three-pointers made, free throws made, rebounds, assists, steals, blocks, and other pertinent data.
# 
# Basketball analysts, team managers, and fantasy sports fans who want to make educated decisions about player performance can find this predictive model useful. The model has the potential to aid in strategic decision-making, player rotation, and fantasy team selection by providing accurate player point predictions.

# ### Preprocessing Data: 
# 
# 

# In[2]:


import pandas as pd

# Load the dataset
game_details = pd.read_csv('games_details.csv')

# Check for missing values
for col in game_details:
    print(col, game_details[col].isnull().sum()/game_details.shape[0], "%")

# Handle missing values
# Example: Impute missing numerical values with mean
#game_details.fillna(game_details.mean(), inplace=True)

# Explore the distribution of numerical features
#print("Summary Statistics:\n", game_details.describe())


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


# In[3]:


# In[15]:


# Example: One-hot encoding for categorical variables
# game_details_encoded = pd.get_dummies(game_details, columns=['PLAYER_NAME', 'START_POSITION'])



# In[ ]:


#from sklearn.model_selection import train_test_split

# Specify features (X) and target variable (y)
# X = game_details_encoded.drop('PTS', axis=1)
# y = game_details_encoded['PTS']

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# from sklearn.linear_model import LinearRegression

# Create a linear regression model
# model = LinearRegression()

# Train the model
# model.fit(X_train, y_train)

# Make predictions on the test set
# predictions = model.predict(X_test)


# In[ ]:


# from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the model
# mse = mean_squared_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)

# print("Mean Squared Error:", mse)
# print("R-squared:", r2)


# In[8]:


# import matplotlib.pyplot as plt
# import seaborn as sns

# Distribution of Points
# plt.figure(figsize=(8, 6))
# sns.histplot(df['PTS'], kde=True)
# plt.title('Distribution of Points Scored')
# plt.xlabel('Points')
# plt.show()


# In[9]:


# correlation_matrix = df.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.show()


# ### What's Done so far?
# Preprocessing of Data:
# 
# Filled in the blanks with zeros to address the missing values.
# Label encoding was used to encode categorical variables (team abbreviations).
# Divide the dataset into sets for testing and training.
# 
# Training and Assessment of Models:
# 
# Due to its initial simplicity, a basic linear regression model was selected.
# Used the training dataset to train the model.
# Based on the test dataset, made predictions.
# Used important metrics like Mean Squared Error (MSE) and R-squared to assess the model's performance.
# 

# ### Conflicts
# 
# Model Performance Issues: 
# A simple linear regression model was selected for the MVP in order to keep things simple. Remaining tasks include exploring and fine-tuning more complex models.

# ### Next Steps
# 
# Feature Development:
# Examine other features that could enhance the predictive model.
# 
# Fine-tuning the model:
# Try out various regression models and tweak the hyperparameters to get optimal results.
# 
# Documentation:
# Summarize the approach, findings, and recommendations in a comprehensive report.

# In[ ]:




