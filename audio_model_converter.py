import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
from scipy.stats import mode

def extract_audio_features_with_noise(audio, sr, noise_factor=0.005):
    # Add random noise to the audio
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise

    # Proceed with standard feature extraction
    mfccs = librosa.feature.mfcc(y=augmented_audio, sr=sr, n_mfcc=40)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    spectral_centroid = librosa.feature.spectral_centroid(y=augmented_audio, sr=sr)
    stft = np.abs(librosa.stft(augmented_audio))
    stft_mean = np.mean(stft, axis=1)

    combined = np.concatenate([
        np.mean(mfccs.T, axis=0),
        np.mean(mfccs_delta.T, axis=0),
        np.mean(mfccs_delta2.T, axis=0),
        np.mean(spectral_centroid.T, axis=0),
        stft_mean
    ])

    return combined

# Segment a single audio file and extract features
def split_single_audio_file(file_path, segment_duration=1, overlap_duration=0.5, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    samples_per_segment = int(segment_duration * target_sr)
    samples_per_overlap = int(overlap_duration * target_sr)
    total_samples = len(audio)

    segments = []
    short_file_count = 0

    # Handle short audio files
    if total_samples < samples_per_segment:
        print(f"Warning: {file_path} is shorter than the segment duration. Taking the whole audio.")
        short_file_count += 1
        combined_features = extract_audio_features_with_noise(audio, sr)
        segments.append(combined_features)
        return segments

    # Regular segmentation with overlapping
    for start in range(0, total_samples - samples_per_segment + 1, samples_per_segment - samples_per_overlap):
        end = start + samples_per_segment
        segment = audio[start:end]
        combined_features = extract_audio_features_with_noise(segment, sr)
        segments.append(combined_features)

    return segments

# Normalize Features
def normalize_features(features, technique="standard"):
    if technique == "standard":
        scaler = StandardScaler()
    elif technique == "minmax":
        scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled

# Process a Single Audio File and Create CSV for Prediction
def process_single_audio_for_prediction(file_path, segment_duration=1, overlap_duration=0.5, output_csv='single_audio_features.csv'):
    segments = split_single_audio_file(file_path, segment_duration=segment_duration, overlap_duration=overlap_duration)

    if segments:  # Ensure segments list is not empty
        n_features = len(segments[0])  # Number of features per segment
        columns = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(segments, columns=columns)

        # Normalize features
        df.iloc[:, :] = normalize_features(df.iloc[:, :])

        # Save to CSV for prediction
        df.to_csv(output_csv, index=False)
        print(f"Single audio features saved to {output_csv}.")
    else:
        print(f"No valid segments created for {file_path}.")

# Majority Voting for Final Prediction
def majority_voting_prediction(model, csv_file):
    segment_features = pd.read_csv(csv_file)

    # Reshape the data to fit the model (3D array required for CNN)
    X_segments = np.expand_dims(segment_features.values, axis=2)

    # Predict the probabilities for each segment
    segment_predictions = model.predict(X_segments)

    # Get the class predicted for each segment
    predicted_classes = np.argmax(segment_predictions, axis=1)

    # Use majority voting to determine the final label
    final_predicted_label, count = mode(predicted_classes)

    # Safely access the final predicted label
    final_predicted_label_value = final_predicted_label.item() if final_predicted_label.size > 0 else None

    if final_predicted_label_value is None:
        print("Error: Mode returned no values.")
        return None

    # Print the final predicted label for the entire audio file
    print(f"Final Predicted Label for the Audio: {final_predicted_label_value}")
    return final_predicted_label_value  # Return as scalar

def audio_model_converter(audio_file: str):    
    single_audio_file = os.path.join("uploads", audio_file)
    output_csv = 'single_audio_test.csv'

    # Process the single audio file and create the CSV for model prediction
    process_single_audio_for_prediction(single_audio_file, segment_duration=1, overlap_duration=0.5, output_csv=output_csv)

    try:
        # Load the trained model
        model = load_model('FinalModel-V2.keras')  # Path to your saved model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Make predictions using majority voting
    return majority_voting_prediction(model, output_csv)
