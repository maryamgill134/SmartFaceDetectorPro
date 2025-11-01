# 🧠 Smart Face Detector Pro

Smart Face Detector Pro is a **real-time facial emotion recognition and detection system** that combines **Deep Learning**, **Computer Vision**, and **Web Development**.  
Built using **Flask, TensorFlow, and OpenCV**, this project detects **faces, eyes, smiles, and human emotions** live through the webcam and logs results automatically.

---

## 🚀 Overview

This project integrates a **custom-trained Convolutional Neural Network (CNN)** on the **FER2013 dataset** to classify human emotions such as:
> Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral

It also features a complete **Flask web dashboard** for live monitoring, automatic **snapshot saving**, and **report generation** with history logs.

---

## 🎯 Features

- **Real-Time Detection**
  - Detects faces, eyes, and smiles using OpenCV Haar cascades  
  - Recognizes emotions through the trained CNN model (`.h5`)

- **Web Dashboard**
  - Flask-based interface for live video feed  
  - Responsive UI built with HTML, CSS, and JavaScript  

- **Data Management**
  - Saves snapshots and logs into **SQLite database**  
  - Supports exporting detection history as **CSV reports**

- **AI Model**
  - Trained on FER2013 dataset using TensorFlow & Keras  
  - Implements Conv2D, MaxPooling, and Dropout layers  
  - Uses data augmentation for better model generalization

---

## 🧩 Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Flask (Python) |
| **AI/ML** | TensorFlow, Keras |
| **Computer Vision** | OpenCV (Haar Cascades) |
| **Database** | SQLite |
| **Dataset** | FER2013 (Facial Emotion Recognition Dataset) |

---

## 🏗️ Project Structure

```
Smart-Face-Detector-Pro/
│
├── static/                # CSS, JS, and image assets
├── templates/             # HTML templates for Flask
├── models/
│   └── emotion_model.h5   # Trained CNN model
├── database/
│   └── history.db         # SQLite database for detection logs
├── app.py                 # Main Flask application
├── haarcascades/          # XML files for face, eyes, smile detection
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup


```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask Application
```bash
python app.py
```

### 5️⃣ Open in Browser
Visit:  
```
http://127.0.0.1:5000
```

---

## 📊 Model Training (Optional)

To retrain the model using the FER2013 dataset:

```python
# train_model.py (example)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model, data loading, and training logic here
```

Save the trained model as:
```
models/emotion_model.h5
```

---

## 📸 Output Preview

- **Live Webcam Detection**
- **Emotion Overlay (Happy, Sad, etc.)**
- **Auto-saved Snapshots**
- **Detection History Dashboard**
- **CSV Report Export**

---

## 🧠 Key Learnings

- Building and training **CNNs** for emotion recognition  
- Applying **Computer Vision (OpenCV)** for face and feature detection  
- Integrating **Deep Learning models** into a **Flask web app**  
- Managing and exporting data with **SQLite & CSV**

---

## 🏅 Future Enhancements

- Add **user authentication** for personalized history tracking  
- Improve UI with **React or Streamlit**  
- Integrate **cloud storage** for saving snapshots  
- Deploy on **Heroku or AWS**

---

## 📚 Dataset Reference

**FER2013** — Facial Expression Recognition 2013  
Available at: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 💡 Author

**Maryam Fazal**  
AI & Computer Vision Enthusiast | Deep Learning Developer  
📧maryamgill134@gmail.com  
🌐 [LinkedIn Post link](https://www.linkedin.com/posts/maryam-fazal-gill-4401a7269_machinelearning-deeplearning-computervision-activity-7376610706797793280-a19V?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAEGzt7ABgmkRzL2jh6-UBjidUSq4iPll298)
