import cv2
import time
import os
import streamlit as st
from deepface import DeepFace
import numpy as np
from tempfile import NamedTemporaryFile

# ----------> Streamlit User Interface 
st.title("Emotion Based Thumbnail Extractor From Video")

# ----------> Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# ----------> Select emotion
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
selected_emotion = st.selectbox("Select an emotion to extract frames", emotions)

# ----------> Number of thumbnails
max_frames_to_extract = st.slider("Number of thumbnails to extract", 1, 5, 1)

if uploaded_file:
    temp_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    # ----------> Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    valid_frames = []

    # ----------> Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend="opencv")
                detected_emotion = result[0]['dominant_emotion']
                
                if detected_emotion == selected_emotion:
                    valid_frames.append((frame_count, frame.copy()))
            except Exception as e:
                st.error(f"Error analyzing frame {frame_count}: {e}")
    
    cap.release()

    if valid_frames:
        selected_frames = valid_frames[::max(1, len(valid_frames) // max_frames_to_extract)][:max_frames_to_extract]
        output_folder = "extracted_frames_and_Thumbnails"

        os.makedirs(output_folder, exist_ok=True)

        st.success(f"{len(selected_frames)} frames extracted!")

        for i, (frame_number, frame) in enumerate(selected_frames):
            filename = f"{output_folder}/frame_{selected_emotion}_{frame_number}.jpg"
            thumbnail_filename = f"{output_folder}/thumbnail_{selected_emotion}_{frame_number}.jpg"
            
            cv2.imwrite(filename, frame)
            cv2.imwrite(thumbnail_filename, cv2.resize(frame, (150, 150)))

            with open(thumbnail_filename, "rb") as file:
                st.image(file.read(), caption=f"Thumbnail { i+1 }", use_container_width =False)
                st.download_button(label=f"Download Thumbnail { i+1 }", data=file, file_name=f"thumbnail_{selected_emotion}_{frame_number}.jpg", mime="image/jpeg")
    else:
        st.warning("No frames with the selected emotion found!")
