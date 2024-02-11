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
import os
import shutil
from moviepy.editor import VideoFileClip
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone

model1 = YOLO('models/burn.pt')
model2 = YOLO('models/blood.pt')
model3 = YOLO('models/wound.pt')

class_files = ['labels/burn.txt', 'labels/blood.txt', 'labels/wound.txt']
class_lists = []
for class_file in class_files:
    with open(class_file, 'r') as file:
        data = file.read()
        class_lists.append(data.split('\n'))
models = [model1, model2, model3]

# print(class_lists)

FRAME_LENGTH = 2048
HOP_LENGTH = 512

def perform_detection(frame, model, class_list):
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
    return frame

def perform_detection(frame, model, class_list):
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
    return frame

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

@app.post('/video')
async def upload_video(file: UploadFile = File(...)):
    try:
        # Create a directory to save uploaded videos if it doesn't exist
        if not os.path.exists("uploaded_videos"):
            os.makedirs("uploaded_videos")
        
        # Save the uploaded video file
        file_path = os.path.join("uploaded_videos", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # audio extraction
        # folder_path = 'voice/'
        # video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        # print(video_files)

        video_clip = VideoFileClip(file_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(f"audio/{file.filename}.wav")
        video_clip.close()
        y, sr = librosa.load(f"audio/{file.filename}.wav", sr=None, mono=True)
        print(y.shape)
        raw_audio = AudioSegment.from_file(f"audio/{file.filename}.wav")
        samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
        # print('hello')
        trimmed, _ = librosa.effects.trim(samples, top_db=25)
        if len(trimmed) < 180000:
            padded = np.pad(trimmed, (0, 180000-len(trimmed)), 'constant')
        else:
            padded = trimmed[:180000]
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
        print(X.shape)
            # features = librosa.feature.mfcc(y=padded, sr=sr, n_mfcc=40)
            # features_processed = np.mean(features.T, axis=0).reshape(1, -1)
            
        # # Predict the emotion
        emotion_label = decode(np.argmax(speech_emotion_model.predict(X)[0]))
        
        counter = [0, 0, 0]
        cap = cv2.VideoCapture(file_path)
        dir_path = 'detections/' + file.filename
        os.makedirs(dir_path, exist_ok=True)
        blood = False
        burn = False
        wound = False
        count = 0
        while count < 200:
            count += 1
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 500))

            # Variables to track detections
            model2_detected = False
            model3_detected = False

            pred1 = model1.predict(frame)
            pred2 = model2.predict(frame)
            pred3 = model3.predict(frame)
            # model1_detected = pred1[0].boxes.data.any() if len(pred1) >= 1 else False 
            # model2_detected = pred2[0].boxes.data.any() if len(pred2) >= 1 else False
            # model3_detected = pred3[0].boxes.data.any() if len(pred3) >= 1 else False
            if counter[0] < 5 and len(pred1) > 0:
                frame = perform_detection(frame, model1, class_lists[0])
                cv2.imwrite( dir_path + f'/model1-{counter[0]}.jpg', frame)
                print('Image saved')
                counter[0] += 1
            if counter[1] < 5 and len(pred2) > 0:
                frame = perform_detection(frame, model2, class_lists[1])
                cv2.imwrite( dir_path + f'/model2-{counter[1]}.jpg', frame)
                print('Image saved')
                counter[1] += 1 
            if counter[2] < 5 and len(pred3) > 0:
                frame = perform_detection(frame, model3, class_lists[2])
                cv2.imwrite( dir_path + f'/model3-{counter[2]}.jpg', frame)
                print('Image saved')
                counter[2] += 1

            if counter[0] > 0:
                burn = True
            if counter[1] > 0:
                blood = True
            if counter[2] > 0:
                wound = True

            for j in range(7):
                ret, frame = cap.read()
                if not ret:
                    break

        cap.release()
        cv2.destroyAllWindows()
        return {'emotion_label': emotion_label, 'blood': blood, 'wound': wound, 'burn': burn}

            # # Perform detection with each model based on the detected class
            # for model, class_file in zip(models, class_files):
            #     with open(class_file, "r") as my_file:
            #         data = my_file.read()
            #         class_list = data.split("\n")

            #     results = model.predict(frame)
            #     a = results[0].boxes.data
            #     px = pd.DataFrame(a).astype("float")

            #     if not px.empty:
            #         # Update detection status for model 2 and model 3
            #         if model == model2:
            #             model2_detected = True
            #         elif model == model3:
            #             model3_detected = True
                    
                # frame = perform_detection(frame, model, class_list)

            # # If model 2 is detected, model 3 can also be detected simultaneously
            # if model2_detected or model3_detected:
            #     with open(class_files[2], "r") as my_file:
            #         data = my_file.read()
            #         class_list = data.split("\n")
                
            #     results = model3.predict(frame)
            #     print('Model 3 results:', results)
            #     a = results[0].boxes.data
            #     px = pd.DataFrame(a).astype("float")

            #     if not px.empty:
            #         frame = perform_detection(frame, model3, class_list)



        
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)
    finally:
        pass



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
