import librosa
import numpy as np

class AudioProcessor:
    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path)

        duration_ms = int((len(y) / sr) * 1000)

        features = {}
        features['duration_ms'] = duration_ms
        features['danceability'] = self.calculate_danceability(y, sr)
        features['energy'] = np.mean(librosa.feature.rms(y=y))
        features['tempo'] = librosa.beat.tempo(y=y, sr=sr)[0]
        features['valence'] = self.calculate_valence(y, sr)
        features['acousticness'] = self.calculate_acousticness(y, sr)
        features['loudness'] = np.mean(librosa.amplitude_to_db(librosa.feature.rms(y=y)))
        features['instrumentalness'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) / 10000

        return features

    def calculate_danceability(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        return np.mean(onset_env)

    def calculate_valence(self, y, sr):
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        return np.mean(spectral_centroids) / 10000

    def calculate_acousticness(self, y, sr):
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        return np.mean(harmonic) / (np.mean(percussive) + 1e-6)
