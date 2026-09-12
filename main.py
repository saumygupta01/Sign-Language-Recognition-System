"""
main.py  —  Real-Time Sign Language Detection
==============================================

This is the **main application** of the project.  It opens the webcam,
detects the user's hand via MediaPipe, extracts and normalises the 21
hand landmarks, feeds them to the trained Random Forest classifier,
and overlays the predicted ASL letter with confidence on the live
video feed.

Usage
-----
    python main.py

Controls
--------
    SPACE       Add the currently detected letter to the text buffer.
    BACKSPACE   Remove the last character from the text buffer.
    C           Clear the entire text buffer.
    Q / ESC     Quit the application.

Prerequisites
-------------
    1.  Collect training data :  python collect_data.py
    2.  Train the model       :  python train_model.py
"""

# ──────────────────────────── Imports ────────────────────────────
import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import joblib

# ──────────────────────────── Configuration ──────────────────────
MODEL_FILE = os.path.join("models", "sign_language_model.pkl")
LABELS_FILE = os.path.join("models", "label_classes.npy")
HAND_MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# Confidence threshold — predictions below this are shown as "?"
CONFIDENCE_THRESHOLD = 0.70

# Prediction smoothing — a raw prediction must repeat this many
# consecutive frames before the on-screen letter updates. This is a
# debounce, not a majority-vote window, so it reacts as soon as you
# hold a new sign for STABILITY_FRAMES frames instead of waiting for
# an old sign to get outvoted in a long history buffer.
STABILITY_FRAMES = 4

# MediaPipe Hands settings
MAX_NUM_HANDS = 1
MIN_DETECTION_CONF = 0.7
MIN_TRACKING_CONF = 0.5

# Colours (BGR)
COL_PANEL_BG = (25, 25, 25)
COL_ACCENT = (0, 255, 200)       # Cyan / teal
COL_WHITE = (255, 255, 255)
COL_GREY = (180, 180, 180)
COL_RED = (60, 60, 255)
COL_GREEN = (80, 220, 80)
COL_TEXT_BG = (40, 40, 40)

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
    if os.path.exists(HAND_MODEL_PATH):
        return
    print(f"[INFO] Downloading hand landmarker model to '{HAND_MODEL_PATH}' …")
    import urllib.request
    os.makedirs(os.path.dirname(HAND_MODEL_PATH), exist_ok=True)
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/latest/"
           "hand_landmarker.task")
    urllib.request.urlretrieve(url, HAND_MODEL_PATH)
    print("[INFO] Model downloaded successfully.")


def normalise_landmarks(hand_landmarks):
    """
    Normalise 21 MediaPipe hand landmarks relative to the wrist
    (landmark 0) AND scale by hand size (distance from wrist to the
    middle-finger MCP, landmark 9), returning a flat 63-element
    feature vector.

    This MUST match the normalisation used in collect_data.py so
    that the model receives features in the same format it was
    trained on.
    """
    wrist = hand_landmarks[0]
    wx, wy, wz = wrist.x, wrist.y, wrist.z

    ref = hand_landmarks[9]
    scale = np.sqrt((ref.x - wx) ** 2 + (ref.y - wy) ** 2 + (ref.z - wz) ** 2)
    if scale < 1e-6:
        scale = 1e-6

    features = []
    for lm in hand_landmarks:
        features.extend([(lm.x - wx) / scale, (lm.y - wy) / scale, (lm.z - wz) / scale])

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
        radius = 6 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, (px, py), radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)


def get_hand_bbox(hand_landmarks, frame_w, frame_h, padding=20):
    """
    Compute a bounding box around the detected hand landmarks.
    """
    x_coords = [lm.x * frame_w for lm in hand_landmarks]
    y_coords = [lm.y * frame_h for lm in hand_landmarks]

    x_min = int(max(min(x_coords) - padding, 0))
    y_min = int(max(min(y_coords) - padding, 0))
    x_max = int(min(max(x_coords) + padding, frame_w))
    y_max = int(min(max(y_coords) + padding, frame_h))

    return x_min, y_min, x_max, y_max


def draw_rounded_rect(img, pt1, pt2, colour, radius=15, thickness=-1):
    """Draw a rectangle with rounded corners using OpenCV primitives."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Clamp radius
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Four corner circles
    cv2.circle(img, (x1 + r, y1 + r), r, colour, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, colour, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, colour, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, colour, thickness)

    # Fill rectangles between circles
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), colour, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), colour, thickness)


def draw_hud(frame, predicted_sign, confidence, fps, text_buffer,
             hand_detected):
    """
    Draw a polished heads-up display (HUD) on the video frame.
    """
    h, w, _ = frame.shape

    # ── Top bar ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, "REAL-TIME SIGN LANGUAGE DETECTION",
                (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                COL_ACCENT, 2, cv2.LINE_AA)

    # FPS on the right
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 130, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COL_GREEN, 2, cv2.LINE_AA)

    # ── Prediction panel (bottom-left) ──
    panel_x, panel_y = 15, h - 170
    panel_w, panel_h = 300, 155
    overlay2 = frame.copy()
    draw_rounded_rect(overlay2,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      COL_PANEL_BG, radius=12)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Hand status
    if hand_detected:
        status_text = "Hand Detected"
        status_col = COL_GREEN
    else:
        status_text = "No Hand Detected"
        status_col = COL_RED

    cv2.putText(frame, status_text,
                (panel_x + 15, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                status_col, 1, cv2.LINE_AA)

    # Predicted sign
    if predicted_sign and hand_detected:
        # Large letter
        cv2.putText(frame, predicted_sign,
                    (panel_x + 15, panel_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    COL_WHITE, 4, cv2.LINE_AA)

        # Confidence bar
        bar_x = panel_x + 100
        bar_y = panel_y + 65
        bar_w = 180
        bar_h = 18
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        fill = int(bar_w * confidence)
        bar_col = COL_GREEN if confidence >= CONFIDENCE_THRESHOLD else COL_RED
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h),
                      bar_col, -1)
        cv2.putText(frame, f"{confidence * 100:.1f}%",
                    (bar_x + bar_w + 5, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COL_GREY, 1, cv2.LINE_AA)

        # Label
        cv2.putText(frame, f"Sign: {predicted_sign}  "
                           f"Conf: {confidence*100:.1f}%",
                    (panel_x + 15, panel_y + 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COL_ACCENT, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "—",
                    (panel_x + 15, panel_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    COL_GREY, 3, cv2.LINE_AA)
        cv2.putText(frame, "Waiting for gesture …",
                    (panel_x + 15, panel_y + 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COL_GREY, 1, cv2.LINE_AA)

    # ── Text buffer panel (bottom-right) ──
    if text_buffer:
        buf_text = "".join(text_buffer)
        buf_w = max(250, len(buf_text) * 18 + 40)
        buf_x = w - buf_w - 15
        buf_y = h - 80

        overlay3 = frame.copy()
        draw_rounded_rect(overlay3,
                          (buf_x, buf_y), (w - 15, h - 15),
                          COL_PANEL_BG, radius=10)
        cv2.addWeighted(overlay3, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "Text:",
                    (buf_x + 10, buf_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COL_GREY, 1, cv2.LINE_AA)
        cv2.putText(frame, buf_text,
                    (buf_x + 10, buf_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COL_WHITE, 2, cv2.LINE_AA)

    # ── Controls hint (very bottom) ──
    cv2.putText(frame, "SPACE:add | BACKSPACE:del | C:clear | Q:quit",
                (15, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (120, 120, 120), 1, cv2.LINE_AA)


# ──────────────────────────── Main ───────────────────────────────

def main():
    print("=" * 55)
    print("  REAL-TIME SIGN LANGUAGE DETECTION")
    print("=" * 55)

    # ── Download hand model if needed ──
    download_model_if_needed()

    # ── Load the trained ML model ──
    if not os.path.exists(MODEL_FILE):
        print(f"\n[ERROR] Trained model not found at '{MODEL_FILE}'.")
        print("        Please run  train_model.py  first.")
        sys.exit(1)

    model = joblib.load(MODEL_FILE)
    print(f"[INFO] Model loaded from '{MODEL_FILE}'")

    if os.path.exists(LABELS_FILE):
        labels = np.load(LABELS_FILE, allow_pickle=True).tolist()
        print(f"[INFO] Classes: {labels}")
    else:
        labels = None

    # ── Initialise MediaPipe Hand Landmarker (new Tasks API) ──
    base_options = mp_python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
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
        print("\n[ERROR] Could not access the webcam.")
        print("        Please check your camera and try again.")
        sys.exit(1)

    print("[INFO] Webcam opened. Press Q or ESC to quit.\n")

    # ── State variables ──
    text_buffer = []            # For the optional word-formation feature
    prev_time = time.time()
    fps = 0.0
    last_stable_sign = None     # For word-formation: avoid adding duplicates
    frame_timestamp_ms = 0      # Monotonically increasing for VIDEO mode

    # Debounce state for prediction stability
    pending_sign = None         # Raw prediction we're currently "counting"
    pending_count = 0           # How many consecutive frames it has repeated
    displayed_sign = None       # What's actually shown on screen right now

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            # Flip horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape

            # Convert BGR → RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # ── Hand detection ──
            frame_timestamp_ms += 33  # ~30 FPS
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            predicted_sign = None
            confidence = 0.0
            hand_detected = False

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                hand_detected = True

                # Draw landmarks
                draw_hand_landmarks(frame, hand_landmarks)

                # Draw bounding box
                x1, y1, x2, y2 = get_hand_bbox(hand_landmarks,
                                                frame_w, frame_h)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              COL_ACCENT, 2)

                # ── Feature extraction & prediction ──
                features = normalise_landmarks(hand_landmarks)
                features_array = np.array(features).reshape(1, -1)

                # Get class probabilities from the Random Forest
                probabilities = model.predict_proba(features_array)[0]
                max_idx = np.argmax(probabilities)
                confidence = probabilities[max_idx]

                if labels:
                    raw_prediction = labels[max_idx]
                else:
                    raw_prediction = model.classes_[max_idx]

                # ── Debounce (smoothing) ──
                # Only count a frame toward stability if it clears the
                # confidence bar; otherwise treat it like a "no vote"
                # frame that doesn't reset progress on its own but
                # also doesn't count toward switching letters.
                if confidence >= CONFIDENCE_THRESHOLD:
                    if raw_prediction == pending_sign:
                        pending_count += 1
                    else:
                        pending_sign = raw_prediction
                        pending_count = 1

                    if pending_count >= STABILITY_FRAMES:
                        displayed_sign = pending_sign

                predicted_sign = displayed_sign

            else:
                # No hand → reset debounce state and clear the display
                pending_sign = None
                pending_count = 0
                displayed_sign = None
                predicted_sign = None

            # ── FPS calculation ──
            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = current_time

            # ── Draw the HUD ──
            draw_hud(frame, predicted_sign, confidence, fps,
                     text_buffer, hand_detected)

            # Show the frame
            cv2.imshow("Sign Language Detection", frame)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF

            # Q or ESC → quit
            if key == ord('q') or key == 27:
                break

            # SPACE → add the current prediction to the text buffer
            if key == 32:  # space
                if predicted_sign and predicted_sign != last_stable_sign:
                    text_buffer.append(predicted_sign)
                    last_stable_sign = predicted_sign
                elif predicted_sign is None:
                    text_buffer.append(" ")
                    last_stable_sign = None

            # BACKSPACE → remove last character
            if key == 8:  # backspace
                if text_buffer:
                    text_buffer.pop()
                    last_stable_sign = None

            # C → clear the text buffer
            if key == ord('c') or key == ord('C'):
                text_buffer.clear()
                last_stable_sign = None

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    if text_buffer:
        print(f"\n[INFO] Final text: {''.join(text_buffer)}")
    print("[INFO] Application closed. Goodbye!")


# ──────────────────────────── Entry Point ────────────────────────
if __name__ == "__main__":
    main()
