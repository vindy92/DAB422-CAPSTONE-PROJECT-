import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Step 1: Load the dataset
try:
    df = pd.read_csv('songs.csv', encoding='ISO-8859-1')  # Try using ISO-8859-1 encoding
except UnicodeDecodeError:
    df = pd.read_csv('songs.csv', encoding='utf-16')

df = df.dropna()

label_encoder_artist = LabelEncoder()
label_encoder_country = LabelEncoder()
label_encoder_genre = LabelEncoder()
label_encoder_durationms = LabelEncoder()
label_encoder_speechiness = LabelEncoder()
label_encoder_energy = LabelEncoder()
label_encoder_valence = LabelEncoder()
label_encoder_danceability = LabelEncoder()
label_encoder_acousticness = LabelEncoder()
label_encoder_tempo = LabelEncoder()
label_encoder_liveness = LabelEncoder()

df['artist_name'] = label_encoder_artist.fit_transform(df['artist_name'])
df['country_name'] = label_encoder_country.fit_transform(df['country_name'])
df['artist_genres'] = label_encoder_genre.fit_transform(df['artist_genres'])
df['duration_ms'] = label_encoder_durationms.fit_transform(df['duration_ms'])
df['speechiness'] = label_encoder_speechiness.fit_transform(df['speechiness'])
df['energy'] = label_encoder_energy.fit_transform(df['energy'])
df['valence'] = label_encoder_valence.fit_transform(df['valence'])
df['danceability'] = label_encoder_danceability.fit_transform(df['danceability'])
df['acousticness'] = label_encoder_acousticness.fit_transform(df['acousticness'])
df['tempo'] = label_encoder_tempo.fit_transform(df['tempo'])
df['liveness'] = label_encoder_liveness.fit_transform(df['liveness'])

selected_features = ['artist_name', 'country_name', 'artist_genres', 'duration_ms', 
                     'speechiness', 'energy', 'valence', 'danceability', 'acousticness', 
                     'tempo', 'liveness']

X = df[selected_features]
y = df['track_popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Success')

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder_artist.pkl', 'wb') as f:
    pickle.dump(label_encoder_artist, f)
with open('label_encoder_country.pkl', 'wb') as f:
    pickle.dump(label_encoder_country, f)
with open('label_encoder_genre.pkl', 'wb') as f:
    pickle.dump(label_encoder_genre, f)
with open('label_encoder_durationms.pkl', 'wb') as f:
    pickle.dump(label_encoder_durationms, f)
with open('label_encoder_speechiness.pkl', 'wb') as f:
    pickle.dump(label_encoder_speechiness, f)
with open('label_encoder_energy.pkl', 'wb') as f:
    pickle.dump(label_encoder_energy, f)
with open('label_encoder_valence.pkl', 'wb') as f:
    pickle.dump(label_encoder_valence, f)
with open('label_encoder_danceability.pkl', 'wb') as f:
    pickle.dump(label_encoder_danceability, f)
with open('label_encoder_acousticness.pkl', 'wb') as f:
    pickle.dump(label_encoder_acousticness, f)
with open('label_encoder_tempo.pkl', 'wb') as f:
    pickle.dump(label_encoder_tempo, f)
with open('label_encoder_liveness.pkl', 'wb') as f:
    pickle.dump(label_encoder_liveness, f)
    
def predict_track_popularity(artist_name, country_name, artist_genres, duration_ms,
                          speechiness, energy, valence, danceability, acousticness,
                          tempo, liveness):

    input_data = pd.DataFrame({
        'artist_name': [artist_name],
        'country_name': [country_name],
        'artist_genres': [artist_genres],
        'duration_ms': [duration_ms],
        'speechiness': [speechiness],
        'energy': [energy],
        'valence': [valence],
        'danceability': [danceability],
        'acousticness': [acousticness],
        'tempo': [tempo],
        'liveness': [liveness],
    })

    input_data['artist_name'] = label_encoder_artist.transform(input_data['artist_name'])
    input_data['country_name'] = label_encoder_country.transform(input_data['country_name'])
    input_data['artist_genres'] = label_encoder_genre.transform(input_data['artist_genres'])
    input_data['duration_ms'] = label_encoder_durationms.fit_transform(input_data['duration_ms'])
    input_data['speechiness'] = label_encoder_speechiness.fit_transform(input_data['speechiness'])
    input_data['energy'] = label_encoder_energy.fit_transform(input_data['energy'])
    input_data['valence'] = label_encoder_valence.fit_transform(input_data['valence'])
    input_data['danceability'] = label_encoder_danceability.fit_transform(input_data['danceability'])
    input_data['acousticness'] = label_encoder_acousticness.fit_transform(input_data['acousticness'])
    input_data['tempo'] = label_encoder_tempo.fit_transform(input_data['tempo'])
    input_data['liveness'] = label_encoder_liveness.fit_transform(input_data['liveness'])

    # Use the trained model to predict the track popularity
    predicted_popularity = model.predict(input_data)
    print(f'Mean Absolute Error: {predicted_popularity}')
    return predicted_popularity[0]
