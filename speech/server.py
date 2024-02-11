from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import librosa
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import tempfile
from pydub import AudioSegment, effects

FRAME_LENGTH = 2048
HOP_LENGTH = 512

emotion_dic = {
    'neutral' : 0,
    'happy'   : 1,
    'sad'     : 2,
    'angry'   : 3,
    'fear'    : 4,
    'disgust' : 5
}
emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust' ]

def encode(label):
    return emotion_dic.get(label)

def decode(category):
    return emotions[category]

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

speech_emotion_model = load_model('model1.h5')
print(speech_emotion_model.summary())

@app.post("/upload")
async def upload_wav_file(file: UploadFile = File(...)):
    # Check if the uploaded file is a WAV file
    if not file.filename.endswith(".wav"):
        return JSONResponse(status_code=400, content={"error": "Only WAV files are allowed"})
    
    with open('temp.wav', "wb") as buffer:
        buffer.write(await file.read())
    
    # Perform feature extraction
    y, sr = librosa.load('temp.wav', sr=None, mono=True)
    raw_audio = AudioSegment.from_file('temp.wav')
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, 180000-len(trimmed)), 'constant')
    zcr = librosa.feature.zero_crossing_rate(padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=padded, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    zcr_list = [zcr]
    rms_list = [rms]
    mfccs_list = [mfccs]
    X = np.concatenate((
        np.swapaxes(zcr_list, 1, 2),
        np.swapaxes(rms_list, 1, 2),
        np.swapaxes(mfccs_list, 1, 2)),
        axis=2
    )
    X = X.astype('float32')

        # features = librosa.feature.mfcc(y=padded, sr=sr, n_mfcc=40)
        # features_processed = np.mean(features.T, axis=0).reshape(1, -1)
        
    # # Predict the emotion
    emotion_label = decode(np.argmax(speech_emotion_model.predict(X)[0]))
    
    return {'emotion': emotion_label}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
