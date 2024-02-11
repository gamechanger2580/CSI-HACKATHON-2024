import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from moviepy.editor import VideoFileClip
import os

# Initialize YOLO models for different classes
model1 = YOLO('models/burn.pt')
model2 = YOLO('models/blood.pt')
model3 = YOLO('models/wound.pt')

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

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

folder_path = 'voice/'

video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
print(video_files)

if video_files:
    video_file = os.path.join(folder_path, video_files[0])

    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(os.path.join(folder_path, "extracted_audio.wav"))
    video_clip.close()

    print("Audio extracted successfully from:", video_file)
else:
    print("No .mp4 files found in the directory:", folder_path)

# Initialize video capture with the selected video file
cap = cv2.VideoCapture(video_file)

# Open class-specific text files
class_files = ['labels/burn.txt', 'labels/blood.txt', 'labels/wound.txt']
models = [model1, model2, model3]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Variables to track detections
    model2_detected = False
    model3_detected = False

    # Perform detection with each model based on the detected class
    for model, class_file in zip(models, class_files):
        with open(class_file, "r") as my_file:
            data = my_file.read()
            class_list = data.split("\n")

        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        if not px.empty:
            # Update detection status for model 2 and model 3
            if model == model2:
                model2_detected = True
            elif model == model3:
                model3_detected = True
            
            frame = perform_detection(frame, model, class_list)

    # If model 2 is detected, model 3 can also be detected simultaneously
    if model2_detected and not model3_detected:
        with open(class_files[2], "r") as my_file:
            data = my_file.read()
            class_list = data.split("\n")
        
        results = model3.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        if not px.empty:
            frame = perform_detection(frame, model3, class_list)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
