from flask import Flask, request, jsonify
import cv2
from utils import ActivityDetector

app = Flask(__name__)
detector = ActivityDetector()

@app.route("/analyze", methods=["POST"])
def analyze_video():
    file = request.files["video"]
    path = "temp.mp4"
    file.save(path)

    cap = cv2.VideoCapture(path)
    result = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        activity, reps, height, speed = detector.process(frame)
        result = {
            "activity": activity,
            "reps": reps,
            "jump_height_cm": height,
            "jump_speed_sec": round(speed, 2)
        }

    cap.release()
    return jsonify(result)

@app.route("/health")
def health():
    return {"status": "running"}

if __name__ == "__main__":
    app.run(debug=True)
