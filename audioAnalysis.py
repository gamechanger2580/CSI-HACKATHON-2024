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
    if len(pause_durations) > 0:
        mean_pause_duration = np.mean(pause_durations)
    else:
        mean_pause_duration = 0

    # Calculate speech rate (words per minute)
    # Assuming average word length is 5 characters
    num_words = len(text.split())
    speech_rate = (num_words / duration) * 60 if duration > 0 else 0

    return mean_pitch, mean_intensity, mean_zcr, duration, mean_pause_duration, speech_rate

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
    mean_pitch, mean_intensity, mean_zcr, duration, mean_pause_duration, speech_rate = extract_acoustic_features(audio_data, sr)

    # You can define your own heuristic or machine learning model to estimate anxiety level
    # For simplicity, let's use a basic formula
    anxiety_score = (mean_pitch + mean_intensity + mean_zcr) * duration / (mean_pause_duration + 1)  # A simple formula

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

        # Check speech rate
        print("Speech rate (words per minute):", extract_acoustic_features(audio_np, sr)[-1])

    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print("Error fetching results; {0}".format(e))
