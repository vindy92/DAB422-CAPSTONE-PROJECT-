import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load the fitted label encoders
label_encoder_artist = pickle.load(open('label_encoder_artist.pkl', 'rb'))
label_encoder_country = pickle.load(open('label_encoder_country.pkl', 'rb'))
label_encoder_genre = pickle.load(open('label_encoder_genre.pkl', 'rb'))
label_encoder_durationms = pickle.load(open('label_encoder_durationms.pkl', 'rb'))
label_encoder_speechiness = pickle.load(open('label_encoder_speechiness.pkl', 'rb'))
label_encoder_energy = pickle.load(open('label_encoder_energy.pkl', 'rb'))
label_encoder_valence = pickle.load(open('label_encoder_valence.pkl', 'rb'))
label_encoder_danceability = pickle.load(open('label_encoder_danceability.pkl', 'rb'))
label_encoder_acousticness = pickle.load(open('label_encoder_acousticness.pkl', 'rb'))
label_encoder_tempo = pickle.load(open('label_encoder_tempo.pkl', 'rb'))
label_encoder_liveness = pickle.load(open('label_encoder_liveness.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Collect data from the form (adjust field names accordingly)
        artist_name = request.form['artist_name']
        country = request.form['country']
        genre = request.form['genre']
        duration_ms = float(request.form['duration_ms'])
        speechiness = float(request.form['speechiness'])
        energy = float(request.form['energy'])
        valence = float(request.form['valence'])
        danceability = float(request.form['danceability'])
        acousticness = float(request.form['acousticness'])
        tempo = float(request.form['tempo'])
        liveness = float(request.form['liveness'])

        # Encode categorical values using the loaded LabelEncoders
        artist_name_encoded = label_encoder_artist.transform([artist_name])[0]
        country_encoded = label_encoder_country.transform([country])[0]
        genre_encoded = label_encoder_genre.transform([genre])[0]
        durationms_encoded = label_encoder_durationms.transform([duration_ms])[0]
        speechiness_encoded = label_encoder_speechiness.transform([speechiness])[0]
        energy_encoded = label_encoder_energy.transform([energy])[0]
        valence_encoded = label_encoder_valence.transform([valence])[0]
        danceability_encoded = label_encoder_danceability.transform([danceability])[0]
        acousticness_encoded = label_encoder_acousticness.transform([acousticness])[0]
        tempo_encoded = label_encoder_tempo.transform([tempo])[0]
        liveness_encoded = label_encoder_liveness.transform([liveness])[0]

        # Prepare the features for prediction (same order as training)
        features = np.array([[artist_name_encoded, country_encoded, genre_encoded, durationms_encoded, speechiness_encoded, energy_encoded, valence_encoded,
                                danceability_encoded, acousticness_encoded, tempo_encoded, liveness_encoded]])

        # Predict popularity using the trained model
        popularity_prediction = model.predict(features)

        # Return the prediction result to the template
        return render_template('prediction.html', prediction=popularity_prediction[0])

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
