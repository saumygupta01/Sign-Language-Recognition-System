# 🤟 Real-Time Sign Language Recognition System

A real-time computer vision and machine learning application that recognizes hand gestures representing American Sign Language (ASL) alphabet letters using a standard webcam.

The system uses **MediaPipe Hand Landmarker** to detect 21 hand landmarks, converts them into a normalized 63-feature representation, and uses a **Random Forest classifier** to recognize the trained sign classes.

The project also includes a **PyQt5 desktop GUI** for live detection, dataset collection, model training, and dataset management.

---

## 📌 Overview

The project provides a lightweight approach to recognizing static hand gestures in real time. Instead of feeding complete camera images directly into a deep-learning model, the system extracts the geometric structure of the hand using MediaPipe and classifies the resulting numerical features.

```text
Webcam
   ↓
OpenCV Frame Capture
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
Normalization
   ↓
63 Numerical Features
   ↓
Random Forest Classifier
   ↓
Confidence Filtering
   ↓
Prediction Stability / Debouncing
   ↓
Detected Letter
   ↓
Text / Word Formation
```

---

# 🎯 Objectives

1. Detect a user's hand in real time using a webcam.
2. Extract the 21 hand landmarks provided by MediaPipe.
3. Convert hand landmarks into a compact numerical feature vector.
4. Normalize the features to reduce the effect of hand position and camera distance.
5. Create a custom dataset of sign-language gestures.
6. Train a machine-learning classifier to recognize gesture classes.
7. Evaluate the trained model using multiple classification metrics.
8. Perform real-time sign prediction.
9. Reduce unstable predictions using confidence filtering and temporal stability.
10. Allow detected letters to be combined into words.
11. Provide a graphical interface for the complete system.

---

# 🧠 Current Machine Learning Approach

## Random Forest Classifier

The current implementation uses a **Random Forest Classifier**, not a CNN.

Random Forest is an ensemble machine-learning algorithm consisting of multiple decision trees.

Configuration:

```text
Number of trees (n_estimators) = 100
Maximum tree depth             = None
Random state                   = 42
CPU usage                      = All available cores
```

### Why Random Forest?

- The input is structured numerical landmark data.
- The feature vector contains only 63 values.
- It works well with small-to-medium tabular datasets.
- It does not require a GPU.
- It provides class probability estimates.
- It is lightweight for real-time inference.
- It does not require neural-network backpropagation.

---

# ✋ Hand Landmark Detection

The project uses **MediaPipe Hand Landmarker**.

MediaPipe provides:

```text
21 landmarks × 3 coordinates (X, Y, Z)
= 63 numerical features
```

The current implementation processes one hand.

Configuration:

```text
Maximum hands                = 1
Minimum detection confidence = 0.70
Minimum tracking confidence  = 0.50
```

Collected features are stored in:

```text
dataset/landmarks.csv
```

---

# 📐 Feature Extraction and Normalization

Raw landmark coordinates depend on the hand's position and distance from the camera.

The project therefore performs two normalization steps.

## 1. Translation Normalization

The wrist is used as the reference point:

```text
x' = x - wrist_x
y' = y - wrist_y
z' = z - wrist_z
```

This makes the representation less dependent on where the hand appears in the frame.

## 2. Scale Normalization

The distance between the wrist and the middle-finger MCP joint (landmark 9) is used as a scale reference.

```text
x_normalized = (x - wrist_x) / scale
y_normalized = (y - wrist_y) / scale
z_normalized = (z - wrist_z) / scale
```

The final representation contains:

```text
63 normalized values
```

The same normalization is used during collection and real-time prediction.

---

# 📊 Dataset

The dataset is stored in:

```text
dataset/landmarks.csv
```

Each row contains:

```text
63 normalized features + 1 class label
```

Conceptually:

```text
feat_0
feat_1
feat_2
...
feat_62
label
```

The data collector is configured for up to:

```text
200 samples per class
```

---

# 🗂️ Dataset Collection

Run:

```bash
python collect_data.py
```

### Controls

```text
A-Z       Select the gesture class
SPACE     Start / stop recording
1 / ESC   Exit
```

The collector automatically stops when the configured number of samples for a class has been reached.

Existing data can also be loaded so collection can be resumed.

---

# 🗑️ Delete a Class

To remove an incorrectly recorded class:

```bash
python delete_class.py <CLASS_LABEL>
```

Example:

```bash
python delete_class.py B
```

This removes samples for that class from:

```text
dataset/landmarks.csv
```

---

# 🔄 Rescale an Older Dataset

If an older dataset was created before scale normalization was introduced, run:

```bash
python rescale_dataset.py
```

This migrates the older feature representation to the current scale-normalized format.

---

# 🏋️ Model Training

Run:

```bash
python train_model.py
```

Training performs:

```text
Load landmarks.csv
       ↓
Separate features and labels
       ↓
Create training/testing sets
       ↓
Train Random Forest
       ↓
Generate predictions
       ↓
Calculate evaluation metrics
       ↓
Generate confusion matrix
       ↓
Save trained model
```

The input contains 63 features and the final CSV column is the class label.

---

# 📚 Train/Test Split

The configured test size is:

```text
20%
```

Approximately:

```text
80% → Training
20% → Testing
```

The current implementation performs the split separately for each class. Earlier samples are used for training and the later samples are held out for testing.

This reduces the problem of placing nearly identical consecutive webcam frames into both sets.

However, this is still not equivalent to testing on a completely independent recording session. A stronger evaluation would use separate recording sessions with different distances, angles, and lighting.

---

# 🌲 Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
```

| Parameter | Value | Meaning |
|---|---:|---|
| `n_estimators` | 100 | Number of decision trees |
| `max_depth` | None | No manually specified maximum depth |
| `random_state` | 42 | Reproducible results |
| `n_jobs` | -1 | Uses available CPU cores |

Training is performed using:

```python
model.fit(X_train, y_train)
```

---

# 📈 Model Evaluation

The model calculates:

### Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how often predictions for a class are correct.

### Recall

Measures how many actual samples of a class are correctly detected.

### F1 Score

Combines precision and recall.

The training script reports macro-averaged precision, recall, and F1 score and also produces a per-class classification report.

---

# 🔥 Confusion Matrix

A confusion matrix is generated after evaluation and saved to:

```text
models/confusion_matrix.png
```

It compares actual classes against predicted classes.

Values on the diagonal represent correct classifications, while off-diagonal values represent misclassifications.

---

# 💾 Saved Model Files

After training:

```text
models/
├── sign_language_model.pkl
├── label_classes.npy
└── confusion_matrix.png
```

### `sign_language_model.pkl`

Contains the trained Random Forest classifier.

### `label_classes.npy`

Stores the ordered class labels used by the model.

### `confusion_matrix.png`

Contains the model's confusion-matrix visualization.

---

# 🎥 Real-Time Detection

After training, run:

```bash
python main.py
```

The application:

1. Opens the webcam.
2. Detects the hand using MediaPipe.
3. Extracts 21 landmarks.
4. Normalizes the landmarks.
5. Creates a 63-feature vector.
6. Sends the vector to the trained Random Forest.
7. Obtains class probabilities.
8. Selects the highest-probability class.
9. Applies confidence filtering.
10. Applies prediction stability.
11. Displays the predicted letter.

The trained model is loaded from:

```text
models/sign_language_model.pkl
```

---

# 🎯 Confidence Filtering

The system uses:

```text
CONFIDENCE_THRESHOLD = 0.70
```

A prediction must reach at least 70% confidence before it contributes to the displayed prediction.

The Random Forest probabilities are obtained using:

```python
model.predict_proba(features_array)
```

---

# ⏱️ Prediction Stability

To reduce unstable predictions caused by individual frames, the system uses:

```text
STABILITY_FRAMES = 4
```

A high-confidence prediction must repeat for four consecutive frames before the displayed letter is updated.

This is a debounce/stability mechanism.

---

# 📝 Word Formation

The real-time application provides a text buffer.

```text
SPACE       Add current detected letter
BACKSPACE   Delete previous character
C           Clear the text
Q / ESC     Quit
```

This allows recognized letters to be combined into words.

---

# 🖥️ Graphical User Interface

Launch the complete desktop application with:

```bash
python gui_app.py
```

The PyQt5 interface contains four tabs:

```text
1. Live Detection
2. Data Collection
3. Train Model
4. Manage Dataset
```

The GUI reuses the same computer-vision and normalization logic and runs the existing training and dataset scripts rather than duplicating their logic.

## Live Detection

Provides:

- Webcam feed
- Hand detection status
- Predicted letter
- Confidence bar
- Text buffer
- Add letter
- Backspace
- Clear
- Start/Stop camera

## Data Collection

Allows users to:

- Select a gesture class
- Start the webcam
- Record samples
- View sample counts
- Save collected data

## Train Model

Runs:

```bash
python train_model.py
```

and displays training output and the generated confusion matrix.

## Manage Dataset

Provides:

- Class sample counts
- Delete class
- Rescale legacy dataset
- Refresh dataset information

---

# 📁 Project Structure

```text
Sign-Language-Recognition-System/
│
├── dataset/
│   └── landmarks.csv
│
├── models/
│   ├── hand_landmarker.task
│   ├── sign_language_model.pkl
│   ├── label_classes.npy
│   └── confusion_matrix.png
│
├── collect_data.py
├── delete_class.py
├── rescale_dataset.py
├── train_model.py
├── main.py
├── gui_app.py
├── Requirement.txt
├── README.md
└── project_report.md
```

| File | Purpose |
|---|---|
| `collect_data.py` | Collects hand landmark training samples |
| `delete_class.py` | Removes a selected class |
| `rescale_dataset.py` | Migrates older dataset features |
| `train_model.py` | Trains and evaluates Random Forest |
| `main.py` | Runs real-time sign recognition |
| `gui_app.py` | Complete PyQt5 desktop interface |
| `landmarks.csv` | Normalized training features |
| `sign_language_model.pkl` | Saved Random Forest model |
| `label_classes.npy` | Saved class-label ordering |
| `confusion_matrix.png` | Evaluation visualization |
| `hand_landmarker.task` | MediaPipe hand-landmark model |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/saumygupta01/Sign-Language-Recognition-System.git
cd Sign-Language-Recognition-System
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scriptsctivate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

The dependency file is:

```text
Requirement.txt
```

Install it with:

```bash
pip install -r Requirement.txt
```

Main libraries include:

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib
- PyQt5
- Matplotlib
- Seaborn

---

# 🚀 Quick Start

For a new dataset:

```bash
python collect_data.py
```

Train the classifier:

```bash
python train_model.py
```

Run real-time recognition:

```bash
python main.py
```

Or launch the complete GUI:

```bash
python gui_app.py
```

Recommended workflow:

```text
Collect Data
     ↓
Manage Dataset
     ↓
Train Model
     ↓
Evaluate Model
     ↓
Run Real-Time Detection
```

---

# 🔧 Configuration

| Parameter | Default | Description |
|---|---:|---|
| `SAMPLES_PER_CLASS` | 200 | Maximum samples per class |
| `CONFIDENCE_THRESHOLD` | 0.70 | Minimum prediction confidence |
| `STABILITY_FRAMES` | 4 | Consecutive matching frames required |
| `N_ESTIMATORS` | 100 | Number of Random Forest trees |
| `TEST_SIZE` | 0.20 | Test portion of each class |
| `MAX_NUM_HANDS` | 1 | Maximum hands processed |
| `MIN_DETECTION_CONF` | 0.70 | MediaPipe detection confidence |
| `MIN_TRACKING_CONF` | 0.50 | MediaPipe tracking confidence |

---

# ⚠️ Limitations

### One Hand

The current implementation processes only one detected hand.

### Lighting

Poor lighting can reduce MediaPipe hand-tracking quality and affect classification.

### Static Gestures

The current classifier works with individual hand poses and does not model gestures as temporal sequences.

### Dataset Dependence

Performance depends heavily on dataset quality and diversity. Changes in hand angle, distance, lighting, background, and orientation can affect real-world performance.

### Evaluation

Samples from one continuous recording can be highly similar. Even with the current per-class holdout strategy, evaluation is not equivalent to testing on a completely independent recording session.

### Not a Complete Translator

This is a demonstration and learning project for gesture-class recognition. It is not a complete sentence-level sign-language translation system.

---

# 🛠️ Improvements Already Applied

## Scale-Invariant Features

The feature representation was updated to divide coordinates by the wrist-to-landmark-9 distance, reducing sensitivity to camera distance.

## Faster Prediction Stabilization

The system uses consecutive high-confidence predictions instead of a long majority-vote history, reducing recognition delay.

## Integrated GUI

PyQt5 combines detection, data collection, model training, and dataset management into one desktop application.

---

# 🔮 Future Improvements

- Collect larger and more diverse datasets.
- Record training data across multiple sessions.
- Test on completely unseen recording sessions.
- Improve recognition of visually similar gestures.
- Support two-hand gestures.
- Support dynamic gestures using temporal models.
- Experiment with CNN-based image classification.
- Experiment with LSTM/sequence-based approaches.
- Add more sign-language classes.
- Add numbers and additional signs.
- Add text-to-speech output.
- Create a mobile version.
- Add sentence-level language processing.

A CNN or LSTM would be a **future alternative**, not the current classifier used by this implementation.

---

# 💻 Hardware Requirements

- Laptop or desktop computer
- Working webcam
- At least 4 GB RAM
- Modern CPU
- No dedicated GPU required

The current classifier operates on 63 landmark features rather than raw camera images, making the training and inference pipeline relatively lightweight.

---

# 🧪 Technology Stack

```text
Python
│
├── OpenCV
│   └── Webcam and image processing
│
├── MediaPipe
│   └── Hand detection and 21 landmark extraction
│
├── NumPy
│   └── Numerical operations
│
├── Pandas
│   └── Dataset handling
│
├── Scikit-learn
│   └── Random Forest and evaluation
│
├── Joblib
│   └── Model serialization
│
├── Matplotlib + Seaborn
│   └── Confusion matrix visualization
│
└── PyQt5
    └── Desktop graphical interface
```

---

# 📌 Technical Summary

```text
Input:
Webcam frame

Hand Detection:
MediaPipe Hand Landmarker

Landmarks:
21

Coordinates:
X, Y, Z

Features:
21 × 3 = 63

Preprocessing:
Wrist-relative normalization
+
Scale normalization

Dataset:
CSV

Learning Type:
Supervised Classification

Classifier:
Random Forest

Number of Trees:
100

Training Split:
80%

Testing Split:
20%

Prediction:
Random Forest probability

Confidence Threshold:
70%

Stability:
4 consecutive frames

Saved Model:
sign_language_model.pkl

Saved Labels:
label_classes.npy
```

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgements

- **MediaPipe** — for hand detection and landmark extraction.
- **OpenCV** — for webcam capture and computer vision operations.
- **Scikit-learn** — for Random Forest and evaluation tools.
- **NumPy** — for numerical computation.
- **Pandas** — for dataset handling.
- **PyQt5** — for the desktop graphical interface.
