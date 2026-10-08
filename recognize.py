"""
recognize.py
------------
Loads the trained Random Forest model (models/gesture_model.pkl) and runs
real-time hand gesture recognition on the webcam feed using MediaPipe Hands.

Displays the predicted gesture label and the model's confidence on-screen.

Press 'q' to quit.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import joblib

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "gesture_model.pkl")
NUM_LANDMARKS = 21
CONFIDENCE_THRESHOLD = 0.6  # below this, label as "UNKNOWN"


def load_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run train_model.py first to create it."
        )
    return joblib.load(path)


def extract_features(hand_landmarks):
    """Flattens MediaPipe landmarks into a single (1, 63) feature vector."""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features).reshape(1, -1)


def main():
    try:
        print("[INFO] Loading trained model...")
        model = load_model(MODEL_PATH)
        print("[INFO] Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection/index.")
        return

    print("[INFO] Starting real-time recognition. Press 'q' to quit.")

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:

            while True:
                success, frame = cap.read()
                if not success:
                    print("[WARNING] Failed to read frame from webcam.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                prediction_text = "No hand detected"

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        try:
                            features = extract_features(hand_landmarks)

                            # Guard against malformed feature vectors
                            expected_features = NUM_LANDMARKS * 3
                            if features.shape[1] != expected_features:
                                prediction_text = "Feature extraction error"
                                continue

                            probabilities = model.predict_proba(features)[0]
                            best_idx = int(np.argmax(probabilities))
                            predicted_label = model.classes_[best_idx]
                            confidence = probabilities[best_idx]

                            if confidence >= CONFIDENCE_THRESHOLD:
                                prediction_text = f"{predicted_label} ({confidence * 100:.1f}%)"
                            else:
                                prediction_text = f"UNKNOWN ({confidence * 100:.1f}%)"

                        except Exception as e:
                            prediction_text = "Prediction error"
                            print(f"[WARNING] Prediction failed: {e}")

                # --- Overlay result on frame ---
                cv2.putText(
                    frame, prediction_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, "Press 'q' to quit", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                )

                cv2.imshow("Real-Time Gesture Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Recognition session ended.")


if __name__ == "__main__":
    main()
