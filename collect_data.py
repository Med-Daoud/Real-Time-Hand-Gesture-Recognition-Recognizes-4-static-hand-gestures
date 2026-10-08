"""
collect_data.py
----------------
Captures hand landmarks from a webcam using MediaPipe Hands and saves them
to data/gestures.csv for later training.

Supported gestures (change/extend as needed):
    OPEN, FIST, PEACE, THUMBS_UP

Controls while the webcam window is focused:
    0 -> label frames as OPEN
    1 -> label frames as FIST
    2 -> label frames as PEACE
    3 -> label frames as THUMBS_UP
    s -> start/stop continuous recording for the currently selected label
    q -> quit and save

Each saved row contains 21 landmarks * 3 coordinates (x, y, z) = 63 features,
plus a final 'label' column.
"""

import os
import csv
import cv2
import mediapipe as mp

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "gestures.csv")

GESTURES = {
    ord("0"): "OPEN",
    ord("1"): "FIST",
    ord("2"): "PEACE",
    ord("3"): "THUMBS_UP",
}

NUM_LANDMARKS = 21  # MediaPipe Hands always returns 21 landmarks per hand


def build_csv_header():
    """Builds the CSV header: x0,y0,z0,...,x20,y20,z20,label"""
    header = []
    for i in range(NUM_LANDMARKS):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    return header


def ensure_csv_exists():
    """Creates the data directory and CSV file with header if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(build_csv_header())


def extract_landmark_row(hand_landmarks, label):
    """Flattens a MediaPipe hand landmark object into a single CSV row."""
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    row.append(label)
    return row


def main():
    ensure_csv_exists()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection/index.")
        return

    current_label = None
    recording = False
    sample_count = 0

    print("=" * 60)
    print("Hand Gesture Data Collection")
    print("Press 0=OPEN 1=FIST 2=PEACE 3=THUMBS_UP to select a label")
    print("Press 's' to start/stop recording, 'q' to quit")
    print("=" * 60)

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands, open(CSV_PATH, mode="a", newline="") as f:

            writer = csv.writer(f)

            while True:
                success, frame = cap.read()
                if not success:
                    print("[WARNING] Failed to read frame from webcam.")
                    continue

                frame = cv2.flip(frame, 1)  # mirror for natural interaction
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        if recording and current_label is not None:
                            row = extract_landmark_row(hand_landmarks, current_label)
                            writer.writerow(row)
                            sample_count += 1

                # --- HUD overlay ---
                status_text = f"Label: {current_label or 'NONE'} | Recording: {recording} | Samples: {sample_count}"
                cv2.putText(
                    frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, "0:OPEN 1:FIST 2:PEACE 3:THUMBS_UP  s:rec  q:quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                )

                cv2.imshow("Collect Gesture Data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in GESTURES:
                    current_label = GESTURES[key]
                    print(f"[INFO] Selected label: {current_label}")
                elif key == ord("s"):
                    if current_label is None:
                        print("[WARNING] Select a label (0-3) before recording.")
                    else:
                        recording = not recording
                        print(f"[INFO] Recording {'started' if recording else 'stopped'}.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Data collection finished. Total samples saved this session: {sample_count}")
        print(f"[INFO] Data stored at: {CSV_PATH}")


if __name__ == "__main__":
    main()
