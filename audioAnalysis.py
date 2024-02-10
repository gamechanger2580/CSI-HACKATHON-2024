import numpy as np
import librosa
import speech_recognition as sr

# Function to extract acoustic features from audio
def extract_acoustic_features(audio_data, sr):
    # Convert audio data to floating-point format
    y = audio_data.astype(np.float32)

    # Extract pitch using librosa
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    mean_pitch = np.nanmean(pitches)

    # Extract intensity (loudness) using root mean square (RMS) amplitude
    rms = librosa.feature.rms(y=y)
    mean_intensity = np.mean(rms)

    # Extract zero crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    mean_zcr = np.mean(zcr)

    # Calculate duration of speech
    duration = librosa.get_duration(y=y, sr=sr)

    # Calculate average pause duration (silence duration)
    pauses = librosa.effects.split(y)
    pause_durations = np.diff(pauses) / sr
    MFCCs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_Bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    if len(pause_durations) > 0:
        mean_pause_duration = np.mean(pause_durations)
    else:
        mean_pause_duration = 0

    return mean_pitch, mean_intensity, mean_zcr, duration, mean_pause_duration, MFCCs, spectral_Bandwidth, spectral_centroid

# Function to classify anxiety level
def classify_anxiety_level(anxiety_score):
    if anxiety_score > 100:
        return 'Very High'
    elif anxiety_score > 75:
        return 'High'
    elif anxiety_score > 50:
        return 'Medium'
    elif anxiety_score > 25:
        return 'Low'
    else:
        return 'Calm'

# Function to estimate anxiety level based on acoustic features
def estimate_anxiety_level(audio_data, sr):
    # Extract acoustic features
    mean_pitch, mean_intensity, mean_zcr, duration, mean_pause_duration, MFCCs, spectral_Bandwidth, spectral_centroid = extract_acoustic_features(audio_data, sr)
    print("Mean pitch:", mean_pitch)
    print("Mean intensity:", mean_intensity)
    print("Mean ZCR:", mean_zcr)
    print("Duration:", duration)
    print("Mean pause duration:", mean_pause_duration)
    print("MFCCs:", MFCCs)
    print("Spectral bandwidth:", spectral_Bandwidth)
    print("Spectral centroid:", spectral_centroid)
    
    
    anxiety_score = 0
    # Calculate anxiety score based on acoustic features
    if mean_pitch > 200:
        anxiety_score += 50
    if mean_intensity > 0.1:
        anxiety_score += 50
    if mean_zcr > 0.1:
        anxiety_score += 50
    if duration > 10:
        anxiety_score += 50
    if mean_pause_duration > 0.1:
        anxiety_score += 50
    if np.mean(MFCCs) > 0.1:
        anxiety_score += 50
    if np.mean(spectral_Bandwidth) > 0.1:
        anxiety_score += 50
    if np.mean(spectral_centroid) > 0.1:
        anxiety_score += 50


    return anxiety_score

# Initialize the recognizer
recognizer = sr.Recognizer()

# Record audio from the microphone
with sr.Microphone() as source:
    print("Please speak something to analyze your anxiety level:")
    audio_data = recognizer.listen(source)

    try:
        # Recognize speech using Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)

        # Convert speech to numpy array and sample rate
        audio_np = np.frombuffer(audio_data.frame_data, dtype=np.int16)
        sr = audio_data.sample_rate

        # Call the function to estimate anxiety level
        anxiety_score = estimate_anxiety_level(audio_np, sr)
        print("Anxiety score:", anxiety_score)

        # Classify anxiety level
        anxiety_level = classify_anxiety_level(anxiety_score)
        print("Anxiety level:", anxiety_level)

    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print("Error fetching results; {0}".format(e))

