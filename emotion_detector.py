import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load trained CNN model
model = load_model(os.path.join("model", "emotion_model.h5"))

# Emotion labels (MUST match training order)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    os.path.join("haarcascade", "haarcascade_frontalface_default.xml")
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = roi_gray.reshape(1, 48, 48, 1)

        prediction = model.predict(roi_gray, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
