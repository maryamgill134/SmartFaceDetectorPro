import os
import sys
import time
import sqlite3
import datetime
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify

# --- TensorFlow (optional emotion model) ---
TF_AVAILABLE = False
emotion_detection_model = None
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_enabled = False

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("TensorFlow imported")
except Exception as e:
    print("TensorFlow not available — emotion detection disabled:", e)
    TF_AVAILABLE = False

# --- Paths ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
DB_PATH = os.path.join(APP_ROOT, "database.db")
MODEL_PATH = os.path.join(APP_ROOT, "models", "emotion_detection_model.h5")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(APP_ROOT, "models"), exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Load Haar cascades ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT, "haarcascade_frontalface_default.xml"))
if eye_cascade.empty():
    eye_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT, "haarcascade_eye.xml"))
if smile_cascade.empty():
    smile_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT, "haarcascade_smile.xml"))

if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print("Haarcascade files missing. Download them or check paths.")

# --- Load Emotion Model ---
if TF_AVAILABLE:
    if os.path.exists(MODEL_PATH):
        try:
            emotion_detection_model = load_model(MODEL_PATH)
            emotion_enabled = True
            print("Emotion model loaded")
        except Exception as e:
            print("Failed to load emotion model:", e)
            emotion_enabled = False
    else:
        print("Emotion model file not found at:", MODEL_PATH)
        emotion_enabled = False
else:
    emotion_enabled = False


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            label TEXT,
            filename TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def prune_missing_snapshots():
    """Drop history rows whose image files were deleted so the gallery does not 404."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, filename FROM detections")
    for row_id, filename in cur.fetchall():
        if not filename:
            cur.execute("DELETE FROM detections WHERE id = ?", (row_id,))
            continue
        path = os.path.join(UPLOAD_FOLDER, os.path.basename(filename))
        if not os.path.isfile(path):
            cur.execute("DELETE FROM detections WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


prune_missing_snapshots()


def log_detection(label, filename):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO detections (timestamp, label, filename) VALUES (?, ?, ?)",
        (ts, label, filename),
    )
    conn.commit()
    conn.close()


def get_logs(limit=100):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, timestamp, label, filename FROM detections ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "timestamp": r[1], "label": r[2], "filename": r[3]} for r in rows]


# --- Camera Setup ---
cap = None
cap_initialized = False
cap_lock = threading.Lock()
camera_fail_until = 0.0
camera_enabled = True


def _camera_backends():
    backends = []
    if sys.platform.startswith("win"):
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DirectShow", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("Media Foundation", cv2.CAP_MSMF))
    backends.append(("Default", cv2.CAP_ANY))
    return backends


def stop_camera():
    global cap, cap_initialized, camera_enabled
    camera_enabled = False
    with cap_lock:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
        cap_initialized = False
    return True


def start_camera():
    global camera_enabled, camera_fail_until
    camera_enabled = True
    camera_fail_until = 0.0
    cam = init_camera()
    return cam is not None and cam.isOpened()


def camera_is_running():
    return bool(camera_enabled and cap is not None and cap.isOpened())


def init_camera():
    """Open the webcam once. On Windows, DirectShow is tried first so OpenCV does not hang."""
    global cap, cap_initialized, camera_fail_until

    if not camera_enabled:
        return None

    if cap_initialized and cap is not None and cap.isOpened():
        return cap

    now = time.time()
    if now < camera_fail_until:
        return None

    with cap_lock:
        if not camera_enabled:
            return None
        if cap_initialized and cap is not None and cap.isOpened():
            return cap

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            cap_initialized = False

        for name, backend in _camera_backends():
            candidate = None
            try:
                candidate = cv2.VideoCapture(0, backend)
                if not candidate.isOpened():
                    candidate.release()
                    continue
                candidate.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ok, frame = candidate.read()
                if ok and frame is not None:
                    cap = candidate
                    cap_initialized = True
                    print(f"Camera initialized ({name})")
                    return cap
                candidate.release()
            except Exception as e:
                print(f"Camera backend {name} failed: {e}")
                if candidate is not None:
                    try:
                        candidate.release()
                    except Exception:
                        pass

        print("Camera failed to initialize — using placeholder until retry")
        cap = None
        cap_initialized = False
        camera_fail_until = time.time() + 3.0
        return None


SNAPSHOT_COOLDOWN = 4.0
last_snapshot_time = 0.0
snapshot_lock = threading.Lock()
detect_lock = threading.Lock()

EMOTION_INTERVAL = 0.35
last_emotion_label = None
last_emotion_time = 0.0


def safe_detect(cascade, image, scale_factor, min_neighbors, min_size):
    """Haar classifiers crash if the ROI is too small or used from two threads at once."""
    if cascade is None or cascade.empty() or image is None or getattr(image, "size", 0) == 0:
        return []
    height, width = image.shape[:2]
    if width < min_size[0] or height < min_size[1]:
        return []
    try:
        found = cascade.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        return found
    except cv2.error:
        return []


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
        print("Emotion prediction failed:", e)
        return None


def save_snapshot(frame, label):
    filename = datetime.datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    cv2.imwrite(filepath, frame)
    log_detection(label, filename)
    return filename


def annotate_and_process(frame):
    global last_snapshot_time, last_emotion_label, last_emotion_time

    if frame is None or getattr(frame, "size", 0) == 0:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = safe_detect(face_cascade, gray, 1.1, 5, (60, 60))

    detected_any = False
    label_for_log = None

    for (x, y, w, h) in faces:
        detected_any = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]

        eyes = safe_detect(eye_cascade, roi_gray, 1.1, 6, (15, 15))
        if len(eyes) > 0:
            cv2.putText(frame, "Eyes", (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        smiles = safe_detect(smile_cascade, roi_gray, 1.7, 22, (25, 25))
        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            label_for_log = "Smiling"

        if emotion_enabled:
            now_e = time.time()
            if now_e - last_emotion_time >= EMOTION_INTERVAL:
                predicted = predict_emotion(roi_gray)
                if predicted:
                    last_emotion_label = predicted
                    last_emotion_time = now_e
            if last_emotion_label:
                cv2.putText(
                    frame,
                    last_emotion_label,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 0),
                    2,
                )
                label_for_log = last_emotion_label

        if not label_for_log:
            label_for_log = "Face"

    if detected_any and label_for_log:
        now = time.time()
        with snapshot_lock:
            if now - last_snapshot_time > SNAPSHOT_COOLDOWN:
                save_snapshot(frame.copy(), label_for_log)
                last_snapshot_time = now

    return frame


def _placeholder_frame(title, subtitle):
    frame = np.full((480, 640, 3), 18, dtype=np.uint8)
    cv2.putText(frame, title, (48, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (94, 234, 212), 2)
    cv2.putText(frame, subtitle, (48, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1)
    return frame


def generate_frames():
    while True:
        if not camera_enabled:
            frame = _placeholder_frame("Webcam stopped", "Press S to start  |  Press Q to stop")
            time.sleep(0.2)
        else:
            camera = init_camera()
            frame = None
            if not camera_enabled:
                continue
            if camera is not None and camera.isOpened():
                with cap_lock:
                    success, frame = camera.read()
                if not success or frame is None:
                    time.sleep(0.05)
                    continue
            else:
                frame = _placeholder_frame(
                    "Camera not available",
                    "Check the webcam, then press S to retry",
                )
                time.sleep(0.25)

        if camera_enabled and frame is not None:
            try:
                with detect_lock:
                    annotated = annotate_and_process(frame)
            except Exception as e:
                print("Frame processing failed:", e)
                annotated = frame
        else:
            annotated = frame
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html", emotion_enabled=emotion_enabled)


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logs")
@app.route("/api/logs")
def logs_page():
    return jsonify(get_logs(100))


@app.route("/api/camera/status")
def api_camera_status():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM detections")
    count = cur.fetchone()[0]
    conn.close()
    return jsonify(
        {
            "enabled": camera_enabled,
            "running": camera_is_running(),
            "emotion_enabled": emotion_enabled,
            "latest_emotion": last_emotion_label,
            "detections": count,
        }
    )


@app.route("/api/camera/start", methods=["GET", "POST"])
def api_camera_start():
    running = start_camera()
    return jsonify({"ok": True, "enabled": camera_enabled, "running": running})


@app.route("/api/camera/stop", methods=["GET", "POST"])
def api_camera_stop():
    stop_camera()
    return jsonify({"ok": True, "enabled": False, "running": False})


@app.route("/api/export_csv")
def api_export_csv():
    import io
    import csv

    logs = get_logs(1000)
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["id", "timestamp", "label", "filename"])
    for item in reversed(logs):
        cw.writerow([item["id"], item["timestamp"], item["label"], item["filename"]])
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=detections.csv"},
    )


@app.route("/close")
def close():
    stop_camera()
    return "Webcam closed successfully!"


import atexit


@atexit.register
def cleanup():
    global cap
    try:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    print("Starting Smart Face Detector Pro...")
    print(f"Emotion detection: {'Enabled' if emotion_enabled else 'Disabled'}")
    print("Server running on: http://127.0.0.1:5000")
    # use_reloader=False prevents a second process from grabbing the webcam
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
