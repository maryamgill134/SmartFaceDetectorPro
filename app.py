import os
import time
import sqlite3
import datetime
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify

# --- TensorFlow Import with Better Error Handling ---
TF_AVAILABLE = False
emotion_detection_model = None
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_enabled = False

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("✅ TensorFlow successfully imported")
except ImportError:
    print("❌ TensorFlow not installed. Installing now...")
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0"])
        from tensorflow.keras.models import load_model
        TF_AVAILABLE = True
        print("✅ TensorFlow installed successfully")
    except Exception as e:
        print("❌ Failed to install TensorFlow:", e)
        TF_AVAILABLE = False
except Exception as e:
    print("❌ Error importing TensorFlow:", e)
    TF_AVAILABLE = False

# --- Paths ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
DB_PATH = os.path.join(APP_ROOT, "database.db")
MODEL_PATH = os.path.join(APP_ROOT, "models", "emotion_detection_model.h5")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(APP_ROOT, "models"), exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Load Haar cascades ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Alternative path if OpenCV data not found
if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT, "haarcascade_frontalface_default.xml"))
if eye_cascade.empty():
    eye_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT, "haarcascade_eye.xml"))
if smile_cascade.empty():
    smile_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT, "haarcascade_smile.xml"))

if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print("⚠️ Haarcascade files missing. Download them or check paths.")

# --- Load Emotion Model ---
if TF_AVAILABLE:
    if os.path.exists(MODEL_PATH):
        try:
            emotion_detection_model = load_model(MODEL_PATH)
            emotion_enabled = True
            print("✅ Emotion model loaded successfully")
        except Exception as e:
            print("❌ Failed to load emotion model:", e)
            emotion_enabled = False
    else:
        print("⚠️ Emotion model file not found at:", MODEL_PATH)
        emotion_enabled = False
else:
    print("⚠️ TensorFlow not available - emotion detection disabled")
    emotion_enabled = False

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            label TEXT,
            filename TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_detection(label, filename):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO detections (timestamp, label, filename) VALUES (?, ?, ?)",
                (ts, label, filename))
    conn.commit()
    conn.close()

def get_logs(limit=100):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, label, filename FROM detections ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "timestamp": r[1], "label": r[2], "filename": r[3]} for r in rows]

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

SNAPSHOT_COOLDOWN = 4.0
last_snapshot_time = 0.0
snapshot_lock = threading.Lock()

def predict_emotion(face_gray):
    if not emotion_enabled or emotion_detection_model is None:
        return None
    try:
        img = cv2.resize(face_gray, (48, 48))
        img = img.astype("float32") / 255.0
        img = img.reshape((1, 48, 48, 1))
        preds = emotion_detection_model.predict(img, verbose=0)
        idx = int(np.argmax(preds))
        return EMOTION_LABELS[idx]
    except Exception as e:
        print("⚠️ Emotion prediction failed:", e)
        return None

def save_snapshot(frame, label):
    filename = datetime.datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    cv2.imwrite(filepath, frame)
    log_detection(label, filename)
    return filename

def annotate_and_process(frame):
    global last_snapshot_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    detected_any = False
    label_for_log = None

    for (x, y, w, h) in faces:
        detected_any = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Eyes detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 6, minSize=(15, 15))
        if len(eyes) > 0:
            cv2.putText(frame, "Eyes", (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22, minSize=(25, 25))
        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            label_for_log = "Smiling"

        # Emotion detection
        if emotion_enabled:
            emotion_label = predict_emotion(roi_gray)
            if emotion_label:
                cv2.putText(frame, emotion_label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                label_for_log = emotion_label

        if not label_for_log:
            label_for_log = "Face"

    # Snapshot with cooldown
    if detected_any and label_for_log:
        now = time.time()
        with snapshot_lock:
            if now - last_snapshot_time > SNAPSHOT_COOLDOWN:
                save_snapshot(frame.copy(), label_for_log)
                last_snapshot_time = now

    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        annotated = annotate_and_process(frame)
        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/j/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', emotion_enabled=emotion_enabled)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs_page():
    return jsonify(get_logs(100))

@app.route('/api/export_csv')
def api_export_csv():
    import io
    import csv
    logs = get_logs(1000)
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["id", "timestamp", "label", "filename"])
    for l in reversed(logs):
        cw.writerow([l["id"], l["timestamp"], l["label"], l["filename"]])
    return Response(si.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=detections.csv"})

@app.route('/close')
def close():
    cap.release()
    cv2.destroyAllWindows()
    return "✅ Webcam closed successfully!"

# Cleanup on exit
import atexit
@atexit.register
def cleanup():
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == '__main__':
    print("🚀 Starting Smart Face Detector Pro...")
    print(f"📷 Camera available: {cap.isOpened()}")
    print(f"😊 Emotion detection: {'Enabled' if emotion_enabled else 'Disabled'}")
    print(f"🌐 Server running on: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)