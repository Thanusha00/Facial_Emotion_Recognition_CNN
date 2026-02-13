#  Facial Emotion Recognition System using CNN

##  Project Overview
This project is an end-to-end Facial Emotion Recognition System that detects human emotions from facial expressions using a Convolutional Neural Network (CNN).
The trained deep learning model is integrated with OpenCV to perform real-time emotion detection through webcam input.

---

##  Project Features
- Image preprocessing and normalization
- Data augmentation to improve model generalization
- CNN-based deep learning model for emotion classification
- Real-time facial emotion detection using webcam
- Face detection using Haar Cascade classifier
- Modular and well-structured implementation

---

##  Problem Statement
Understanding human emotions plays a crucial role in human–computer interaction, mental health analysis, and intelligent systems.
This project aims to automatically recognize facial emotions from images using deep learning techniques to enable real-time emotion-aware applications.

---

##  Tech Stack
- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow, Keras  
- **Computer Vision:** OpenCV  
- **Libraries:** NumPy, Matplotlib, Scikit-learn  
- **Face Detection:** Haar Cascade  
- **Tools:** Git, GitHub  

---

##  Dataset
- **FER-2013 Dataset (Kaggle)**
- 48×48 grayscale facial images
- 7 emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

---

##  Project Workflow
1. Dataset Collection  
2. Image Preprocessing & Augmentation  
3. CNN Model Architecture Design  
4. Model Training & Validation  
5. Model Evaluation  
6. Model Saving  
7. Real-time Emotion Detection using OpenCV  

---

##  Model Performance
- Validation Accuracy: **~60–65%**
- Evaluated using:
  - Accuracy
  - Loss Curves
  - Confusion Matrix
  - Classification Report

---

##  Real-Time Emotion Detection
The system captures live video through a webcam, detects faces, and predicts emotions in real time.

**Predicted Emotions:**
- 😠 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😀 Happy  
- 😢 Sad  
- 😲 Surprise  
- 😐 Neutral  

Press **Q** to exit the webcam window.

---

##  How to Run the Project
```bash
pip install -r requirements.txt
python emotion_detector.py
