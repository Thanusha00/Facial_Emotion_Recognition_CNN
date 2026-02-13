#  Facial Emotion Recognition using CNN

##  Project Overview
A deep learning-based system that detects human facial emotions in real-time using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.

---

##  Features
- Image preprocessing & data augmentation
- CNN architecture with Conv2D, MaxPooling, Dropout
- Achieved ~60-65% validation accuracy
- Real-time emotion detection using OpenCV
- Haar Cascade for face detection

---

##  Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

##  Project Structure
```
Facial_Emotion_Recognition/
│
├── model/
│   └── emotion_model.h5
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── emotion_detector.py
├── requirements.txt
└── README.md
```

---

##  How to Run

1. Create virtual environment  
2. Install dependencies  
```
pip install -r requirements.txt
```

3. Run real-time emotion detection  
```
python emotion_detector.py
```

Press **Q** to exit webcam.

---

##  Model Performance
Validation Accuracy: ~60–65%

---

##  Future Improvements
- Improve accuracy using transfer learning
- Deploy as Flask web app
- Add emotion analytics dashboard
