"""
collect_data.py  —  Webcam-Based Hand-Landmark Data Collector
=============================================================

This script opens your webcam, detects your hand using MediaPipe,
extracts 21 hand landmarks (x, y, z), normalises them relative to
the wrist, and saves the resulting 63-feature vectors to a CSV file
along with a class label (A–Z).

Usage
-----
    python collect_data.py

Controls (while the webcam window is active)
---------------------------------------------
    A–Z         Select the gesture class you want to record.
    S           Start / stop recording samples for the selected class.
    Q / ESC     Quit the program.

The collected data is stored in:  dataset/landmarks.csv
"""

# ──────────────────────────── Imports ────────────────────────────
import os
import sys
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ──────────────────────────── Configuration ──────────────────────
# Number of samples to collect per gesture class
SAMPLES_PER_CLASS = 200

# Path where the CSV dataset will be saved
DATASET_DIR = "dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "landmarks.csv")

# Path to the MediaPipe hand-landmarker model file
MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# MediaPipe Hands settings
MAX_NUM_HANDS = 1           # We only track one hand for simplicity
MIN_DETECTION_CONF = 0.7    # Minimum confidence to detect a hand
MIN_TRACKING_CONF = 0.5     # Minimum confidence to keep tracking

# Hand landmark connections for drawing (21 landmarks, 0–20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (5, 9), (9, 13), (13, 17),               # Palm
]


# ──────────────────────────── Helper Functions ───────────────────

def download_model_if_needed():
    """Download the hand_landmarker.task model if it doesn't exist."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"[INFO] Downloading hand landmarker model to '{MODEL_PATH}' …")
    import urllib.request
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/latest/"
           "hand_landmarker.task")
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[INFO] Model downloaded successfully.")


def normalise_landmarks(hand_landmarks):
    """
    Convert 21 MediaPipe hand landmarks into a 63-element feature
    vector that is both **translation-invariant** and
    **scale-invariant**.

    Why normalise like this?
    ─────────────────────────
    1. Translation: subtracting the wrist (landmark 0) makes the
       features independent of where the hand sits in the frame.
    2. Scale: raw wrist-relative coordinates still depend on how far
       the hand is from the camera — a "B" held close to the webcam
       produces much larger numbers than the same "B" held further
       away. Without correcting for this, the model partly learns
       "distance from camera" instead of "shape of the hand", which
       is why predictions get worse/mismatched as soon as you move
       relative to where you collected training data.

       We fix this by dividing every coordinate by the distance
       between the wrist (0) and the middle-finger MCP joint (9).
       That distance is a stable proxy for "how big the hand looks
       in the frame" regardless of finger curl, so dividing by it
       makes the whole feature vector scale-invariant too.

    Parameters
    ----------
    hand_landmarks : list of NormalizedLandmark
        The 21 hand landmarks returned by MediaPipe.

    Returns
    -------
    list[float]
        A flat list of 63 values: [x0, y0, z0, x1, y1, z1, …, x20, y20, z20]
        relative to the wrist and scaled by hand size.
    """
    wrist = hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    # Reference joint for scale: middle finger MCP (landmark 9) is a
    # stable "palm size" proxy that doesn't change much with finger curl.
    ref = hand_landmarks[9]
    scale = np.sqrt(
        (ref.x - wrist_x) ** 2
        + (ref.y - wrist_y) ** 2
        + (ref.z - wrist_z) ** 2
    )
    if scale < 1e-6:
        scale = 1e-6  # guard against division by zero

    features = []
    for lm in hand_landmarks:
        features.extend([
            (lm.x - wrist_x) / scale,
            (lm.y - wrist_y) / scale,
            (lm.z - wrist_z) / scale,
        ])

    return features


def draw_hand_landmarks(frame, hand_landmarks):
    """
    Draw the 21 hand landmarks and connections on the video frame
    using OpenCV (replaces the old mp.solutions.drawing_utils).
    """
    h, w, _ = frame.shape

    # Convert normalised landmarks to pixel coordinates
    points = []
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Draw connections (lines)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx],
                 (0, 255, 0), 2, cv2.LINE_AA)

    # Draw landmark dots
    for i, (px, py) in enumerate(points):
        # Fingertips (4, 8, 12, 16, 20) get a larger dot
        radius = 6 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, (px, py), radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)


def draw_info_panel(frame, current_class, sample_count, is_recording, class_counts):
    """
    Draw an informational overlay on the video frame so the user
    knows which class is selected, how many samples have been
    collected, and whether recording is active.
    """
    h, w, _ = frame.shape

    # Semi-transparent dark panel at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "SIGN LANGUAGE DATA COLLECTOR",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 200), 2, cv2.LINE_AA)

    # Current class + sample count
    status_colour = (0, 0, 255) if is_recording else (200, 200, 200)
    rec_text = "  [RECORDING]" if is_recording else ""
    cv2.putText(frame, f"Class: {current_class}   "
                       f"Samples: {sample_count}/{SAMPLES_PER_CLASS}{rec_text}",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                status_colour, 2, cv2.LINE_AA)

    # Instructions
    cv2.putText(frame, "A-Z: select class | SPACE: start/stop | 1: quit",
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Show per-class counts on the right side
    y_offset = 140
    cv2.putText(frame, "Collected:", (w - 180, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
    y_offset += 25
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        colour = (0, 255, 0) if count >= SAMPLES_PER_CLASS else (200, 200, 200)
        cv2.putText(frame, f"{cls}: {count}",
                    (w - 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    colour, 1, cv2.LINE_AA)
        y_offset += 20
        if y_offset > h - 20:
            break  # Prevent drawing outside the frame


# ──────────────────────────── Main ───────────────────────────────

def main():
    # ── Download model if needed ──
    download_model_if_needed()

    # ── Ensure the dataset directory exists ──
    os.makedirs(DATASET_DIR, exist_ok=True)

    # ── Load any existing data so we can resume collection ──
    class_counts = {}  # e.g. {'A': 150, 'B': 200, ...}
    existing_rows = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                existing_rows.append(row)
                label = row[-1]
                class_counts[label] = class_counts.get(label, 0) + 1
        print(f"[INFO] Loaded {len(existing_rows)} existing samples from {DATASET_FILE}")
        for cls in sorted(class_counts):
            print(f"       {cls}: {class_counts[cls]} samples")
    else:
        header = None

    # ── Prepare CSV header ──
    # 21 landmarks × 3 coordinates = 63 features  +  1 label
    if header is None:
        header = [f"feat_{i}" for i in range(63)] + ["label"]

    # ── Initialise MediaPipe Hand Landmarker (new Tasks API) ──
    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # ── Open the webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        print("        Please check your camera and try again.")
        return

    print("\n[INFO] Webcam opened successfully.")
    print("[INFO] Press A–Z to select a class, SPACE to start/stop recording, 1 to quit.\n")

    current_class = "A"
    is_recording = False
    new_rows = []  # Buffer for newly collected rows
    frame_timestamp_ms = 0  # Monotonically increasing timestamp for VIDEO mode

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            # Flip horizontally so it feels like a mirror
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # ── Hand detection ──
            frame_timestamp_ms += 33  # ~30 FPS
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            hand_detected = False

            if results.hand_landmarks:
                # We only process the first detected hand
                hand_landmarks = results.hand_landmarks[0]
                hand_detected = True

                # Draw the hand landmarks on the frame
                draw_hand_landmarks(frame, hand_landmarks)

                # ── Record sample if recording is active ──
                if is_recording:
                    count = class_counts.get(current_class, 0)
                    if count < SAMPLES_PER_CLASS:
                        features = normalise_landmarks(hand_landmarks)
                        row = [f"{v:.6f}" for v in features] + [current_class]
                        new_rows.append(row)
                        class_counts[current_class] = count + 1
                        print(f"\r       {current_class}: "
                              f"{class_counts[current_class]}/{SAMPLES_PER_CLASS}",
                              end="", flush=True)
                    else:
                        # Enough samples collected — auto-stop
                        is_recording = False
                        print(f"\n[INFO] Finished collecting {SAMPLES_PER_CLASS} "
                              f"samples for class '{current_class}'.")

            # ── Draw the information panel ──
            if not hand_detected:
                cv2.putText(frame, "No hand detected",
                            (15, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)

            draw_info_panel(frame, current_class,
                            class_counts.get(current_class, 0),
                            is_recording, class_counts)

            # Show the frame
            cv2.imshow("Sign Language Data Collector", frame)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF

            # 1 or ESC → quit
            if key == ord('1') or key == 27:
                break

            # SPACE → start / stop recording
            elif key == 32:
                if is_recording:
                    is_recording = False
                    print(f"\n[INFO] Stopped recording for class '{current_class}'.")
                else:
                    count = class_counts.get(current_class, 0)
                    if count >= SAMPLES_PER_CLASS:
                        print(f"\n[INFO] Class '{current_class}' already has "
                              f"{SAMPLES_PER_CLASS} samples. Pick another class.")
                    else:
                        is_recording = True
                        print(f"\n[INFO] Recording samples for class "
                              f"'{current_class}' …")

            # A–Z → select class
            elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
                current_class = chr(key).upper()
                is_recording = False
                print(f"\n[INFO] Selected class: {current_class}  "
                      f"({class_counts.get(current_class, 0)} samples so far)")

    finally:
        # ── Clean up ──
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

        # ── Save collected data ──
        if new_rows:
            with open(DATASET_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                # Write existing rows first, then new rows
                for row in existing_rows:
                    writer.writerow(row)
                for row in new_rows:
                    writer.writerow(row)
            total = len(existing_rows) + len(new_rows)
            print(f"\n[INFO] Saved {total} total samples to {DATASET_FILE}")
        else:
            print("\n[INFO] No new samples were collected.")

    print("[INFO] Data collection finished. Goodbye!")


# ──────────────────────────── Entry Point ────────────────────────
if __name__ == "__main__":
    main()
