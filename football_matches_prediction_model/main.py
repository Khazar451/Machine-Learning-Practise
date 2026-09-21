import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("E0.csv")

df = df[["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].drop_duplicates()
# save only the needed columns
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
# convert str to datetime64[us] for accurate sorting and computer to read it better
df = df.sort_values(by=["Date", "Time"]).reset_index(drop=True)
# we sorted the values for Date and Time then reseted the index
# we resetted the index because it may crash the pipeline for our learning model

team_list = df["HomeTeam"].unique()

df["Home_Rolling3"] = np.nan
df["Away_Rolling3"] = np.nan
# Create empty columns in the master dataset first

for team in team_list:
    team_df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    team_df["TeamGoals"] = np.where(team_df["HomeTeam"] == team, team_df["FTHG"], team_df["FTAG"])
    team_df["TeamGoals_Rolling3"] = team_df["TeamGoals"].rolling(window=3).mean().shift(1)
    # Evaluate the condition on the master df (380 rows), but assign the calculated data from team_df
    df.loc[df["HomeTeam"] == team, "Home_Rolling3"] = team_df["TeamGoals_Rolling3"]
    df.loc[df["AwayTeam"] == team, "Away_Rolling3"] = team_df["TeamGoals_Rolling3"]
df = df.dropna().reset_index(drop=True)
# we drop the NaNs
df["Target"] = df["FTR"].map({"A": 0, "D": 1, "H": 2})
# we change the A, D, H so that the model can read it better

# SKlearn part

# Isolate features (X) and target (y)
X = df[["Home_Rolling3", "Away_Rolling3"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

print(matrix)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


print(df.info())
# print(df[["Date", "HomeTeam", "AwayTeam", "Home_Rolling3", "Away_Rolling3", "Target"]].head())

