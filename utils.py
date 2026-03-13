import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

class ActivityDetector:
    def __init__(self, model_path="models/activity_classifier.joblib"):
        self.model = joblib.load(model_path)
        self.pose = mp.solutions.pose.Pose()
        self.prev_y = None
        self.rep_count = 0
        self.jump_start_time = None
        self.max_jump_height = 0

    def angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def extract_features(self, landmarks):
        shoulder = landmarks[11]
        hip = landmarks[23]
        knee = landmarks[25]

        angle_val = self.angle(
            [shoulder.x, shoulder.y],
            [hip.x, hip.y],
            [knee.x, knee.y]
        )

        vertical_disp = abs(hip.y - knee.y)
        velocity = np.random.uniform(0.1, 0.4)

        return np.array([[angle_val, vertical_disp*100, velocity]])

    def process(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(img)

        activity = "unknown"
        jump_height = 0
        speed = 0

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            features = self.extract_features(lm)
            activity = self.model.predict(features)[0]

            hip_y = lm[23].y

            if self.prev_y is not None:
                delta = self.prev_y - hip_y

                if activity == "jump":
                    if delta > 0.02:
                        self.jump_start_time = self.jump_start_time or time.time()
                        self.max_jump_height = max(self.max_jump_height, delta)
                    elif delta < -0.02 and self.jump_start_time:
                        self.rep_count += 1
                        speed = time.time() - self.jump_start_time
                        jump_height = round(self.max_jump_height * 100, 2)
                        self.jump_start_time = None
                        self.max_jump_height = 0

                elif activity in ["pushup", "situp"]:
                    if delta > 0.015:
                        self.rep_count += 1

            self.prev_y = hip_y

        return activity, self.rep_count, jump_height, speed
