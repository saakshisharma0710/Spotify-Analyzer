import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
print("hello world")
# load dataset
df = pd.read_csv("spotify_tracks.csv")

# select feature columns
selected_cols = [
    'duration_ms',
    'loudness',
    'tempo',
    'energy',
    'acousticness',
    'instrumentalness',
    'danceability',
    'valence'
]

X = df[selected_cols]
y = df['popularity']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
model = RandomForestRegressor(n_estimators=400, random_state=42)

# train
model.fit(X_train, y_train)

# evaluate
score = model.score(X_test, y_test)
print("R² score:", score)

# save model
joblib.dump(model, "popularity_model.pkl")
print("model saved as popularity_model.pkl")
