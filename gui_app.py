"""
gui_app.py — PyQt5 Desktop Application  (GUI, Integration & Overall System)
============================================================================

This file is the "front end" that ties the whole project together into a
single desktop application, instead of everyone running collect_data.py /
main.py / train_model.py / delete_class.py separately from a terminal.

Design choice — this file does NOT modify any of the existing scripts.
It only:
    1. IMPORTS small, already-importable pieces from main.py and
       collect_data.py — config constants and pure helper functions such
       as normalise_landmarks() / draw_hand_landmarks() — so the exact
       same feature-extraction and drawing code the CV/ML side wrote is
       reused instead of copy-pasted.
    2. CALLS train_model.py, delete_class.py and rescale_dataset.py as
       subprocesses (via QProcess), exactly the way you'd run them from
       a terminal, and streams their console output into the GUI.

So every ML/CV file in the repo stays untouched — gui_app.py is purely
the GUI + integration + application-shell layer.

Tabs
----
1. Live Detection    — real-time ASL letter prediction   (GUI version of main.py)
2. Data Collection    — record new samples per letter     (GUI version of collect_data.py)
3. Train Model        — retrain + evaluate the classifier (runs train_model.py)
4. Manage Dataset      — class counts, delete a class, rescale legacy data

Run
---
    python gui_app.py

Requirements
------------
Everything already listed in Requirement.txt (PyQt5, opencv-python,
mediapipe, numpy, scikit-learn, joblib) plus pandas, which train_model.py
and delete_class.py already depend on.
"""

# ──────────────────────────── Imports ────────────────────────────
import os
import sys
import csv

import cv2
import numpy as np
import joblib

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QProcess
from PyQt5.QtGui import QImage, QPixmap, QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QProgressBar, QTextEdit, QTableWidget,
    QTableWidgetItem, QMessageBox, QFrame, QLineEdit, QGroupBox, QShortcut,
    QHeaderView, QSizePolicy,
)
from PyQt5.QtGui import QKeySequence

# Reuse (import — do NOT modify) the config + pure helper functions the
# CV/ML side already wrote. Both modules guard their camera loop behind
# `if __name__ == "__main__":`, so importing them here is safe — it does
# NOT open a webcam or a cv2 window.
import main as detect_core
import collect_data as collect_core

ACCENT = "#00c8a0"          # teal accent, matches the on-screen HUD colour
DARK_BG = "#1a1a1a"
PANEL_BG = "#232323"


# ════════════════════════════════════════════════════════════════════
#  Background worker threads
#
#  Both workers mirror the frame-by-frame logic that already lives in
#  main.py's main() / collect_data.py's main(), just driven by a Qt
#  event loop (QThread + signals) instead of a blocking
#  `while True: cv2.imshow(...); cv2.waitKey(...)` loop, since PyQt and
#  cv2's own window system can't share a GUI thread.
# ════════════════════════════════════════════════════════════════════

def _bgr_frame_to_qimage(frame_bgr):
    """Convert an OpenCV BGR frame to a QImage for display in a QLabel."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class DetectionWorker(QThread):
    """
    Owns the webcam + MediaPipe HandLandmarker + trained model for the
    'Live Detection' tab. Re-implements main.py's per-frame prediction
    and debounce logic (via imported helpers/constants from main.py),
    emitting Qt signals instead of drawing an OpenCV HUD.
    """
    frame_ready = pyqtSignal(QImage)
    prediction_ready = pyqtSignal(str, float, bool)   # letter, confidence, hand_detected
    error = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.model = None
        self.labels = None
        self.landmarker = None
        self.cap = None

    def run(self):
        self._running = True
        try:
            detect_core.download_model_if_needed()

            if not os.path.exists(detect_core.MODEL_FILE):
                self.error.emit(
                    f"No trained model found at '{detect_core.MODEL_FILE}'.\n"
                    "Go to the 'Train Model' tab and train one first."
                )
                return

            self.model = joblib.load(detect_core.MODEL_FILE)
            if os.path.exists(detect_core.LABELS_FILE):
                self.labels = np.load(detect_core.LABELS_FILE, allow_pickle=True).tolist()
            else:
                self.labels = None

            base_options = detect_core.mp_python.BaseOptions(
                model_asset_path=detect_core.HAND_MODEL_PATH
            )
            options = detect_core.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=detect_core.vision.RunningMode.VIDEO,
                num_hands=detect_core.MAX_NUM_HANDS,
                min_hand_detection_confidence=detect_core.MIN_DETECTION_CONF,
                min_tracking_confidence=detect_core.MIN_TRACKING_CONF,
            )
            self.landmarker = detect_core.vision.HandLandmarker.create_from_options(options)

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.error.emit("Could not access the webcam.")
                return

            frame_timestamp_ms = 0
            pending_sign, pending_count, displayed_sign = None, 0, None

            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error.emit("Failed to read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                frame_h, frame_w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = detect_core.mp.Image(
                    image_format=detect_core.mp.ImageFormat.SRGB, data=rgb_frame
                )

                frame_timestamp_ms += 33
                results = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                predicted_sign, confidence, hand_detected = None, 0.0, False

                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    hand_detected = True

                    detect_core.draw_hand_landmarks(frame, hand_landmarks)
                    x1, y1, x2, y2 = detect_core.get_hand_bbox(hand_landmarks, frame_w, frame_h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), detect_core.COL_ACCENT, 2)

                    features = detect_core.normalise_landmarks(hand_landmarks)
                    features_array = np.array(features).reshape(1, -1)
                    probabilities = self.model.predict_proba(features_array)[0]
                    max_idx = np.argmax(probabilities)
                    confidence = float(probabilities[max_idx])
                    raw_prediction = (
                        self.labels[max_idx] if self.labels else self.model.classes_[max_idx]
                    )

                    if confidence >= detect_core.CONFIDENCE_THRESHOLD:
                        if raw_prediction == pending_sign:
                            pending_count += 1
                        else:
                            pending_sign, pending_count = raw_prediction, 1
                        if pending_count >= detect_core.STABILITY_FRAMES:
                            displayed_sign = pending_sign

                    predicted_sign = displayed_sign
                else:
                    pending_sign, pending_count, displayed_sign = None, 0, None

                self.frame_ready.emit(_bgr_frame_to_qimage(frame))
                self.prediction_ready.emit(predicted_sign or "", confidence, hand_detected)

        except Exception as exc:  # noqa: BLE001 — surface any crash to the GUI
            self.error.emit(f"Detection thread crashed: {exc}")
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.landmarker is not None:
                self.landmarker.close()
            self.finished_signal.emit()

    def stop(self):
        self._running = False
        self.wait(2000)


class CollectionWorker(QThread):
    """
    Owns the webcam + MediaPipe HandLandmarker for the 'Data Collection'
    tab. Mirrors collect_data.py's main() loop (via imported helpers from
    collect_data.py), but the GUI toggles the selected class / recording
    state instead of A-Z / SPACE key presses, and saving to CSV is an
    explicit action instead of only happening on exit.
    """
    frame_ready = pyqtSignal(QImage)
    status_ready = pyqtSignal(bool, str, int, int)  # hand_detected, class, count, samples_per_class
    error = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.current_class = "A"
        self.is_recording = False
        self.new_rows = []
        self.class_counts = {}
        self.existing_rows = []
        self.header = None
        self.landmarker = None
        self.cap = None

    def set_class(self, letter):
        self.current_class = letter.upper()
        self.is_recording = False

    def set_recording(self, flag):
        self.is_recording = bool(flag)

    def run(self):
        self._running = True
        try:
            collect_core.download_model_if_needed()
            os.makedirs(collect_core.DATASET_DIR, exist_ok=True)

            if os.path.exists(collect_core.DATASET_FILE):
                with open(collect_core.DATASET_FILE, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        self.existing_rows.append(row)
                        label = row[-1]
                        self.class_counts[label] = self.class_counts.get(label, 0) + 1
            else:
                header = None
            self.header = header or [f"feat_{i}" for i in range(63)] + ["label"]

            base_options = collect_core.mp_python.BaseOptions(
                model_asset_path=collect_core.MODEL_PATH
            )
            options = collect_core.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=collect_core.vision.RunningMode.VIDEO,
                num_hands=collect_core.MAX_NUM_HANDS,
                min_hand_detection_confidence=collect_core.MIN_DETECTION_CONF,
                min_tracking_confidence=collect_core.MIN_TRACKING_CONF,
            )
            self.landmarker = collect_core.vision.HandLandmarker.create_from_options(options)

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.error.emit("Could not access the webcam.")
                return

            frame_timestamp_ms = 0

            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error.emit("Failed to read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = collect_core.mp.Image(
                    image_format=collect_core.mp.ImageFormat.SRGB, data=rgb_frame
                )
                frame_timestamp_ms += 33
                results = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                hand_detected = False
                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    hand_detected = True
                    collect_core.draw_hand_landmarks(frame, hand_landmarks)

                    if self.is_recording:
                        count = self.class_counts.get(self.current_class, 0)
                        if count < collect_core.SAMPLES_PER_CLASS:
                            features = collect_core.normalise_landmarks(hand_landmarks)
                            row = [f"{v:.6f}" for v in features] + [self.current_class]
                            self.new_rows.append(row)
                            self.class_counts[self.current_class] = count + 1
                        else:
                            self.is_recording = False  # auto-stop at the cap

                self.frame_ready.emit(_bgr_frame_to_qimage(frame))
                self.status_ready.emit(
                    hand_detected,
                    self.current_class,
                    self.class_counts.get(self.current_class, 0),
                    collect_core.SAMPLES_PER_CLASS,
                )

        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Collection thread crashed: {exc}")
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.landmarker is not None:
                self.landmarker.close()
            self.finished_signal.emit()

    def stop(self):
        self._running = False
        self.wait(2000)

    def pending_sample_count(self):
        return len(self.new_rows)

    def save_to_csv(self):
        """Append everything recorded this session to dataset/landmarks.csv."""
        if not self.new_rows:
            return 0
        with open(collect_core.DATASET_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            for row in self.existing_rows:
                writer.writerow(row)
            for row in self.new_rows:
                writer.writerow(row)
        saved = len(self.new_rows)
        self.existing_rows.extend(self.new_rows)
        self.new_rows = []
        return saved


# ════════════════════════════════════════════════════════════════════
#  Small shared UI helpers
# ════════════════════════════════════════════════════════════════════

def make_video_label():
    """A consistent placeholder QLabel used for the live camera feed."""
    label = QLabel("Camera is off")
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumSize(560, 420)
    label.setStyleSheet(
        f"background-color: {DARK_BG}; color: #888; border: 1px solid #333; "
        "border-radius: 6px; font-size: 14px;"
    )
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return label


def section_title(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {ACCENT}; font-size: 15px; font-weight: 600;")
    return lbl


# ════════════════════════════════════════════════════════════════════
#  Tab 1 — Live Detection  (GUI front end for main.py)
# ════════════════════════════════════════════════════════════════════

class LiveDetectionTab(QWidget):
    """
    Real-time ASL letter prediction. This is the GUI equivalent of running
    `python main.py`: same model, same normalisation, same debounce logic
    (all imported from main.py) — just rendered as Qt widgets instead of
    an OpenCV HUD, and with buttons standing in for the SPACE / BACKSPACE /
    C key controls.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.text_buffer = []
        self.last_added_sign = None
        self.current_prediction = ""
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── Left: video feed ──
        left = QVBoxLayout()
        self.video_label = make_video_label()
        left.addWidget(self.video_label)

        controls_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        controls_row.addWidget(self.start_btn)
        controls_row.addWidget(self.stop_btn)
        left.addLayout(controls_row)
        root.addLayout(left, 3)

        # ── Right: prediction + text buffer panel ──
        right = QVBoxLayout()
        right.addWidget(section_title("Prediction"))

        self.hand_status_label = QLabel("No hand detected")
        self.hand_status_label.setStyleSheet("color: #aa4444; font-size: 12px;")
        right.addWidget(self.hand_status_label)

        self.letter_label = QLabel("—")
        self.letter_label.setAlignment(Qt.AlignCenter)
        self.letter_label.setStyleSheet(
            f"background-color: {PANEL_BG}; color: white; font-size: 64px; "
            "font-weight: bold; border-radius: 8px; padding: 20px;"
        )
        right.addWidget(self.letter_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setFormat("Confidence: %p%")
        right.addWidget(self.confidence_bar)

        right.addSpacing(10)
        right.addWidget(section_title("Text Buffer"))

        self.text_output = QLineEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setStyleSheet("font-size: 18px; padding: 6px;")
        right.addWidget(self.text_output)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add  (Space)")
        self.add_btn.clicked.connect(self.add_letter)
        self.back_btn = QPushButton("Backspace")
        self.back_btn.clicked.connect(self.backspace)
        self.clear_btn = QPushButton("Clear  (C)")
        self.clear_btn.clicked.connect(self.clear_text)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.back_btn)
        btn_row.addWidget(self.clear_btn)
        right.addLayout(btn_row)

        right.addStretch(1)
        hint = QLabel("Tip: hold a sign steady until it locks in, then click Add.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        right.addWidget(hint)

        root.addLayout(right, 2)

        # Keyboard shortcuts, mirroring main.py's SPACE / BACKSPACE / C controls
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.add_letter)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, activated=self.backspace)
        QShortcut(QKeySequence(Qt.Key_C), self, activated=self.clear_text)

    # ── Worker lifecycle ──
    def start_detection(self):
        if self.worker is not None:
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.video_label.setText("Starting camera…")

        self.worker = DetectionWorker()
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.prediction_ready.connect(self._on_prediction)
        self.worker.error.connect(self._on_error)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def stop_detection(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Camera is off")
        self.video_label.setPixmap(QPixmap())
        self.hand_status_label.setText("No hand detected")
        self.hand_status_label.setStyleSheet("color: #aa4444; font-size: 12px;")
        self.letter_label.setText("—")
        self.confidence_bar.setValue(0)

    # ── Signal handlers ──
    def _on_frame(self, qimage):
        pix = QPixmap.fromImage(qimage).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def _on_prediction(self, letter, confidence, hand_detected):
        self.current_prediction = letter
        if hand_detected:
            self.hand_status_label.setText("Hand detected")
            self.hand_status_label.setStyleSheet("color: #44aa66; font-size: 12px;")
        else:
            self.hand_status_label.setText("No hand detected")
            self.hand_status_label.setStyleSheet("color: #aa4444; font-size: 12px;")

        self.letter_label.setText(letter if letter else "—")
        self.confidence_bar.setValue(int(confidence * 100))

    def _on_error(self, message):
        QMessageBox.warning(self, "Live Detection", message)
        self.stop_detection()

    def _on_finished(self):
        pass  # cleanup already handled in stop_detection()

    # ── Text buffer actions (mirror SPACE / BACKSPACE / C from main.py) ──
    def add_letter(self):
        if not self.isVisible() or self.worker is None:
            return
        if self.current_prediction and self.current_prediction != self.last_added_sign:
            self.text_buffer.append(self.current_prediction)
            self.last_added_sign = self.current_prediction
        elif not self.current_prediction:
            self.text_buffer.append(" ")
            self.last_added_sign = None
        self.text_output.setText("".join(self.text_buffer))

    def backspace(self):
        if not self.isVisible():
            return
        if self.text_buffer:
            self.text_buffer.pop()
            self.last_added_sign = None
        self.text_output.setText("".join(self.text_buffer))

    def clear_text(self):
        if not self.isVisible():
            return
        self.text_buffer.clear()
        self.last_added_sign = None
        self.text_output.setText("")

    def shutdown(self):
        """Called by the main window on app close."""
        if self.worker is not None:
            self.worker.stop()
            self.worker = None


# ════════════════════════════════════════════════════════════════════
#  Shared: run one of the repo's standalone scripts as a subprocess and
#  stream its console output into a QTextEdit. Used by the Train Model
#  and Manage Dataset tabs so train_model.py / delete_class.py /
#  rescale_dataset.py run EXACTLY as written — no logic duplicated,
#  no files touched.
# ════════════════════════════════════════════════════════════════════

class ScriptRunner:
    """Wraps a QProcess that runs `sys.executable <script> [args]` and
    streams stdout/stderr into a QTextEdit log widget."""

    def __init__(self, log_widget: QTextEdit, on_finished=None):
        self.log_widget = log_widget
        self.on_finished = on_finished
        self.process = None

    def is_running(self):
        return self.process is not None and self.process.state() != QProcess.NotRunning

    def run(self, script_name, args=None):
        if self.is_running():
            return
        args = args or []
        self.log_widget.clear()
        self._append(f"$ python {script_name} {' '.join(args)}\n")

        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.start(sys.executable, [script_name] + args)

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append(data)

    def _append(self, text):
        self.log_widget.moveCursor(QTextCursor.End)
        self.log_widget.insertPlainText(text)
        self.log_widget.moveCursor(QTextCursor.End)

    def _finished(self, exit_code, _exit_status):
        self._append(f"\n[process exited with code {exit_code}]\n")
        if self.on_finished:
            self.on_finished(exit_code)


# ════════════════════════════════════════════════════════════════════
#  Tab 3 — Train Model  (runs train_model.py as-is)
# ════════════════════════════════════════════════════════════════════

class TrainModelTab(QWidget):
    """
    Retrains the classifier by running `python train_model.py`, exactly
    as if you'd typed it into a terminal, and shows the resulting
    confusion-matrix image once it finishes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._refresh_matrix()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(section_title("Train / Retrain"))
        info = QLabel(
            "Runs train_model.py: loads dataset/landmarks.csv, trains a "
            "Random Forest, evaluates it, and saves "
            "models/sign_language_model.pkl + models/label_classes.npy."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 11px;")
        left.addWidget(info)

        self.train_btn = QPushButton("Retrain Model")
        self.train_btn.clicked.connect(self.start_training)
        left.addWidget(self.train_btn)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            f"background-color: {DARK_BG}; color: #ddd; font-family: monospace; font-size: 11px;"
        )
        left.addWidget(self.log_output)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        right.addWidget(section_title("Confusion Matrix"))
        self.matrix_label = QLabel("No confusion matrix yet — train the model first.")
        self.matrix_label.setAlignment(Qt.AlignCenter)
        self.matrix_label.setWordWrap(True)
        self.matrix_label.setMinimumSize(340, 300)
        self.matrix_label.setStyleSheet(
            f"background-color: {DARK_BG}; color: #888; border: 1px solid #333; "
            "border-radius: 6px; font-size: 12px;"
        )
        right.addWidget(self.matrix_label)
        right.addStretch(1)
        root.addLayout(right, 2)

        self.runner = ScriptRunner(self.log_output, on_finished=self._on_finished)

    def start_training(self):
        if self.runner.is_running():
            return
        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training…")
        self.runner.run("train_model.py")

    def _on_finished(self, exit_code):
        self.train_btn.setEnabled(True)
        self.train_btn.setText("Retrain Model")
        self._refresh_matrix()
        if exit_code == 0:
            QMessageBox.information(self, "Train Model", "Training finished successfully.")
        else:
            QMessageBox.warning(
                self, "Train Model",
                f"train_model.py exited with code {exit_code}. Check the log for details."
            )

    def _refresh_matrix(self):
        matrix_path = os.path.join("models", "confusion_matrix.png")
        if os.path.exists(matrix_path):
            pix = QPixmap(matrix_path).scaled(
                self.matrix_label.width(), self.matrix_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.matrix_label.setPixmap(pix)
        else:
            self.matrix_label.setText("No confusion matrix yet — train the model first.")


# ════════════════════════════════════════════════════════════════════
#  Tab 4 — Manage Dataset  (runs delete_class.py / rescale_dataset.py as-is)
# ════════════════════════════════════════════════════════════════════

class ManageDatasetTab(QWidget):
    """
    Shows per-class sample counts from dataset/landmarks.csv, and wraps
    delete_class.py / rescale_dataset.py — both run unmodified via
    subprocess, with a confirmation dialog first since both are
    destructive/irreversible operations on the CSV.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.refresh_counts()

    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(section_title("Class Counts (dataset/landmarks.csv)"))

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Class", "Samples"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        left.addWidget(self.table)

        self.refresh_btn = QPushButton("Refresh Counts")
        self.refresh_btn.clicked.connect(self.refresh_counts)
        left.addWidget(self.refresh_btn)
        top.addLayout(left, 2)

        right = QVBoxLayout()
        right.addWidget(section_title("Dataset Tools"))

        del_group = QGroupBox("Delete a class  (delete_class.py)")
        del_layout = QVBoxLayout(del_group)
        del_row = QHBoxLayout()
        self.delete_combo = QComboBox()
        self.delete_btn = QPushButton("Delete Class")
        self.delete_btn.clicked.connect(self.delete_class)
        del_row.addWidget(self.delete_combo)
        del_row.addWidget(self.delete_btn)
        del_layout.addLayout(del_row)
        right.addWidget(del_group)

        rescale_group = QGroupBox("Fix legacy (non scale-invariant) data")
        rescale_layout = QVBoxLayout(rescale_group)
        rescale_info = QLabel(
            "One-time migration (rescale_dataset.py) for datasets recorded "
            "before the scale-invariant normalisation fix."
        )
        rescale_info.setWordWrap(True)
        rescale_info.setStyleSheet("color: #888; font-size: 11px;")
        rescale_layout.addWidget(rescale_info)
        self.rescale_btn = QPushButton("Rescale Dataset")
        self.rescale_btn.clicked.connect(self.rescale_dataset)
        rescale_layout.addWidget(self.rescale_btn)
        right.addWidget(rescale_group)

        right.addStretch(1)
        top.addLayout(right, 1)
        root.addLayout(top)

        root.addWidget(section_title("Log"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        self.log_output.setStyleSheet(
            f"background-color: {DARK_BG}; color: #ddd; font-family: monospace; font-size: 11px;"
        )
        root.addWidget(self.log_output)

        self.runner = ScriptRunner(self.log_output, on_finished=self._on_script_finished)

    def refresh_counts(self):
        counts = {}
        path = collect_core.DATASET_FILE
        if os.path.exists(path):
            try:
                with open(path, "r", newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # header
                    for row in reader:
                        if not row:
                            continue
                        label = row[-1]
                        counts[label] = counts.get(label, 0) + 1
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Manage Dataset", f"Could not read dataset: {exc}")

        self.table.setRowCount(0)
        for label in sorted(counts):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setItem(row, 1, QTableWidgetItem(str(counts[label])))

        self.delete_combo.clear()
        self.delete_combo.addItems(sorted(counts) or ["(no data yet)"])

    def delete_class(self):
        if self.runner.is_running():
            return
        label = self.delete_combo.currentText()
        if not label or label == "(no data yet)":
            return
        reply = QMessageBox.question(
            self, "Delete class",
            f"This permanently removes every recorded sample for class "
            f"'{label}' from landmarks.csv. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._set_tools_enabled(False)
        self.runner.run("delete_class.py", [label])

    def rescale_dataset(self):
        if self.runner.is_running():
            return
        reply = QMessageBox.question(
            self, "Rescale dataset",
            "This rewrites dataset/landmarks.csv in place to make older "
            "samples scale-invariant. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._set_tools_enabled(False)
        self.runner.run("rescale_dataset.py")

    def _set_tools_enabled(self, enabled):
        self.delete_btn.setEnabled(enabled)
        self.rescale_btn.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)

    def _on_script_finished(self, _exit_code):
        self._set_tools_enabled(True)
        self.refresh_counts()


# ════════════════════════════════════════════════════════════════════
#  Main window — ties all four tabs together into one application
# ════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition — Control Center")

        self.detection_tab = LiveDetectionTab()
        self.collection_tab = DataCollectionTab()
        self.train_tab = TrainModelTab()
        self.manage_tab = ManageDatasetTab()

        tabs = QTabWidget()
        tabs.addTab(self.detection_tab, "Live Detection")
        tabs.addTab(self.collection_tab, "Data Collection")
        tabs.addTab(self.train_tab, "Train Model")
        tabs.addTab(self.manage_tab, "Manage Dataset")

        # Stop any running camera worker when the user switches tabs
        # away from it, so only one camera loop ever runs at a time.
        tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs = tabs

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready.")

    def _on_tab_changed(self, index):
        current = self._tabs.widget(index)
        if current is not self.detection_tab:
            self.detection_tab.stop_detection()
        if current is not self.collection_tab:
            self.collection_tab.stop_collection()

    def closeEvent(self, event):
        self.detection_tab.shutdown()
        self.collection_tab.shutdown()
        super().closeEvent(event)


STYLE_SHEET = f"""
QMainWindow, QWidget {{
    background-color: #202020;
    color: #e8e8e8;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid #333;
}}
QTabBar::tab {{
    background: #262626;
    color: #ccc;
    padding: 8px 16px;
}}
QTabBar::tab:selected {{
    background: {PANEL_BG};
    color: {ACCENT};
    font-weight: 600;
}}
QPushButton {{
    background-color: {PANEL_BG};
    color: #e8e8e8;
    border: 1px solid #3a3a3a;
    border-radius: 5px;
    padding: 7px 14px;
}}
QPushButton:hover {{
    border-color: {ACCENT};
}}
QPushButton:disabled {{
    color: #666;
}}
QPushButton:checked {{
    background-color: {ACCENT};
    color: #10221c;
    font-weight: 600;
}}
QProgressBar {{
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    text-align: center;
    background-color: {PANEL_BG};
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
}}
QGroupBox {{
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {ACCENT};
}}
QTableWidget {{
    background-color: {PANEL_BG};
    gridline-color: #3a3a3a;
}}
QHeaderView::section {{
    background-color: #262626;
    color: {ACCENT};
    padding: 4px;
    border: none;
}}
"""


def run_app():
    # Scripts in this project use paths relative to the repo root
    # (e.g. "models/...", "dataset/..."), same as main.py / collect_data.py.
    # Make sure we're always running from the folder this file lives in.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)

    window = MainWindow()
    window.resize(1150, 760)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
# ════════════════════════════════════════════════════════════════════

class DataCollectionTab(QWidget):
    """
    Record new training samples per letter. GUI equivalent of running
    `python collect_data.py`: same MediaPipe pipeline and the same
    normalise_landmarks() feature format (imported from collect_data.py),
    with a class dropdown / Record button standing in for the A-Z / SPACE
    key controls, and an explicit Save button instead of saving only on
    exit.
    """

    LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── Left: video feed ──
        left = QVBoxLayout()
        self.video_label = make_video_label()
        left.addWidget(self.video_label)

        controls_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_collection)
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_collection)
        self.stop_btn.setEnabled(False)
        controls_row.addWidget(self.start_btn)
        controls_row.addWidget(self.stop_btn)
        left.addLayout(controls_row)
        root.addLayout(left, 3)

        # ── Right: recording controls ──
        right = QVBoxLayout()
        right.addWidget(section_title("Record Samples"))

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.LETTERS)
        self.class_combo.currentTextChanged.connect(self._on_class_changed)
        class_row.addWidget(self.class_combo)
        right.addLayout(class_row)

        self.hand_status_label = QLabel("No hand detected")
        self.hand_status_label.setStyleSheet("color: #aa4444; font-size: 12px;")
        right.addWidget(self.hand_status_label)

        self.count_label = QLabel("Samples for A: 0 / 200")
        self.count_label.setStyleSheet(f"color: {ACCENT}; font-size: 14px; font-weight: 600;")
        right.addWidget(self.count_label)

        self.count_progress = QProgressBar()
        self.count_progress.setRange(0, collect_core.SAMPLES_PER_CLASS)
        right.addWidget(self.count_progress)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.setEnabled(False)
        self.record_btn.toggled.connect(self._on_record_toggled)
        right.addWidget(self.record_btn)

        right.addSpacing(10)
        self.pending_label = QLabel("0 new samples not yet saved")
        self.pending_label.setStyleSheet("color: #888; font-size: 12px;")
        right.addWidget(self.pending_label)

        self.save_btn = QPushButton("Save to dataset/landmarks.csv")
        self.save_btn.clicked.connect(self.save_dataset)
        right.addWidget(self.save_btn)

        right.addStretch(1)
        hint = QLabel(
            "Tip: for a model that generalises, record each letter across a "
            "few separate sessions — vary your distance, hand angle and lighting."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        right.addWidget(hint)

        root.addLayout(right, 2)

    # ── Worker lifecycle ──
    def start_collection(self):
        if self.worker is not None:
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.record_btn.setEnabled(True)
        self.video_label.setText("Starting camera…")

        self.worker = CollectionWorker()
        self.worker.set_class(self.class_combo.currentText())
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_ready.connect(self._on_status)
        self.worker.error.connect(self._on_error)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def stop_collection(self):
        if self.worker is not None:
            pending = self.worker.pending_sample_count()
            if pending and self._confirm_discard(pending):
                self.worker.save_to_csv()
            self.worker.stop()
            self.worker = None
        self.record_btn.setChecked(False)
        self.record_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Camera is off")
        self.video_label.setPixmap(QPixmap())
        self.pending_label.setText("0 new samples not yet saved")

    def _confirm_discard(self, pending):
        reply = QMessageBox.question(
            self, "Unsaved samples",
            f"You have {pending} new sample(s) not yet saved to the CSV.\n"
            "Save them before stopping the camera?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    # ── Signal handlers ──
    def _on_frame(self, qimage):
        pix = QPixmap.fromImage(qimage).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def _on_status(self, hand_detected, current_class, count, samples_per_class):
        if hand_detected:
            self.hand_status_label.setText("Hand detected")
            self.hand_status_label.setStyleSheet("color: #44aa66; font-size: 12px;")
        else:
            self.hand_status_label.setText("No hand detected")
            self.hand_status_label.setStyleSheet("color: #aa4444; font-size: 12px;")

        self.count_label.setText(f"Samples for {current_class}: {count} / {samples_per_class}")
        self.count_progress.setValue(count)

        if count >= samples_per_class and self.record_btn.isChecked():
            self.record_btn.setChecked(False)  # auto-stopped by the worker

        if self.worker is not None:
            self.pending_label.setText(
                f"{self.worker.pending_sample_count()} new sample(s) not yet saved"
            )

    def _on_error(self, message):
        QMessageBox.warning(self, "Data Collection", message)
        self.stop_collection()

    def _on_finished(self):
        pass

    # ── Controls ──
    def _on_class_changed(self, letter):
        if self.worker is not None:
            self.worker.set_class(letter)
        self.record_btn.setChecked(False)

    def _on_record_toggled(self, checked):
        if self.worker is not None:
            self.worker.set_recording(checked)
        self.record_btn.setText("Stop Recording" if checked else "Start Recording")

    def save_dataset(self):
        if self.worker is None:
            QMessageBox.information(self, "Data Collection", "Start the camera first.")
            return
        saved = self.worker.save_to_csv()
        self.pending_label.setText("0 new samples not yet saved")
        if saved:
            QMessageBox.information(
                self, "Data Collection", f"Saved {saved} new sample(s) to landmarks.csv."
            )
        else:
            QMessageBox.information(self, "Data Collection", "Nothing new to save.")

    def shutdown(self):
        """Called by the main window on app close."""
        if self.worker is not None:
            pending = self.worker.pending_sample_count()
            if pending and self._confirm_discard(pending):
                self.worker.save_to_csv()
            self.worker.stop()
            self.worker = None
