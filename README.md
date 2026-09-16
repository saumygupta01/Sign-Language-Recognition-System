# Real-Time Sign Language Detection Using Python and Webcam

## Table of Contents
- Abstract
- Problem Statement
- Objectives
- Technologies Used
- System Architecture
- Project Structure
- Installation
- Usage / How to Run
- Methodology
- Machine Learning Algorithm
- Key Features
- Resultsp:[]
- Configuration
- Known Limitations and Fixes Applied
- Future Improvements
- Hardware Requirements
- License
- Acknowledgements

## Abstract

This project is a college-level Computer Vision and Machine Learning application that recognizes American Sign Language (ASL) alphabets in real time using a standard webcam. It uses Google's MediaPipe framework for hand landmark detection, capturing 21 points on the hand, and feeds these into a Random Forest classifier trained to recognize different ASL signs.

The project covers the full pipeline: building a custom dataset, training a model, and running it live to convert hand gestures into text on screen.

## Problem Statement

There is a communication barrier between sign language users and people who do not know sign language. Human interpreters are not always available or affordable. A low-cost, real-time detection system that runs on normal consumer hardware can help bridge this gap.

## Objectives

1. Capture and extract hand landmarks in real time using a webcam.
2. Build a simple interface for creating a custom sign language dataset.
3. Train a classifier to recognize ASL alphabet gestures.
4. Run a real-time prediction pipeline with low latency.
5. Let the user combine detected letters into full words.
6. Keep the system light enough to run on a normal laptop without a GPU.

## Technologies Used

- Python 3.x
- OpenCV - webcam access, frame capture, and display
- MediaPipe - hand detection and landmark extraction
- NumPy - numerical operations
- Pandas - handling the CSV dataset
- Scikit-learn - training and evaluating the Random Forest classifier
- Joblib - saving the trained model
- Matplotlib and Seaborn - plotting the confusion matrix

## System Architecture

```
Webcam
  -> OpenCV (frame capture)
  -> MediaPipe (hand detection, 21 landmarks)
  -> Preprocessing (wrist-relative + scale normalization)
  -> Random Forest classifier
  -> Prediction smoothing (debounce)
  -> Screen display and word formation
```

## Project Structure

```
sign_language_detection/
    dataset/
        landmarks.csv
    models/
        sign_language_model.pkl
        label_classes.npy
        confusion_matrix.png
    collect_data.py
    rescale_dataset.py
    delete_class.py
    train_model.py
    main.py
    requirements.txt
    README.md
    project_report.md
```

`rescale_dataset.py` is a one-time script used to update an older dataset to the current feature format.

`delete_class.py` removes all samples of one class label from the dataset, in case a class was recorded incorrectly.

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection
```

2. Create a virtual environment (recommended):
```
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage / How to Run

### Step 1: Collect data
```
python collect_data.py
```
Controls:
- A-Z: select the class to record
- SPACE: start or stop recording (default 200 samples per class)
- 1 or ESC: quit

For a model that works well outside the exact conditions it was trained in, record each letter across a few separate sessions, changing your distance from the camera, hand angle, and lighting each time. Recording all 200 samples for a letter in one continuous take gives good numbers during evaluation but does not hold up during live use.

### Step 2 (if needed): Remove a bad class
```
python delete_class.py <CLASS_LABEL>
```
Example:
```
python delete_class.py B
```

### Step 2.5 (only for older datasets)
If your dataset was collected before the scale normalization fix, run this once:
```
python rescale_dataset.py
```

### Step 3: Train the model
```
python train_model.py
```
This prints accuracy, precision, recall, and F1 score, and saves a confusion matrix to `models/confusion_matrix.png`. The trained model is saved as `sign_language_model.pkl`.

### Step 4: Run real-time detection
```
python main.py
```
Controls:
- SPACE: add the current letter to the word
- BACKSPACE: delete the last letter
- C: clear the word
- Q or ESC: quit

## Methodology

1. Data collection: the webcam captures frames, MediaPipe detects the hand and extracts 21 landmarks.
2. Feature extraction: each landmark has x, y, z coordinates, giving a 63-value feature vector per frame.
3. Normalization: coordinates are made translation-invariant by subtracting the wrist position, and scale-invariant by dividing by the distance between the wrist and the middle-finger MCP joint (landmark 9). Without the scale step, the same letter produces different feature values depending on distance from the camera, which was causing incorrect predictions.
4. Storage: normalized features and the class label are saved to a CSV file.
5. Model training: the dataset is split per class, taking the earlier recorded samples for training and the later ones for testing, instead of a random shuffle. A random split can mix near-identical consecutive frames between train and test and give an inflated accuracy score.
6. Real-time prediction: live frames are passed to the model, and a prediction needs at least 70 percent confidence to be considered valid.
7. Prediction smoothing: instead of a 10-frame majority vote, the system now waits for a prediction to repeat over a small number of consecutive frames (STABILITY_FRAMES) before updating the letter on screen. This removes most of the delay that came from the old majority-vote approach.

## Machine Learning Algorithm

The project uses a Random Forest classifier.

Random Forest builds many decision trees and outputs the class chosen by most of them. It was chosen because it works well on small tabular datasets, runs without a GPU, gives confidence scores for each prediction, and does not overfit as easily as a single decision tree.

Input: a 63-value, normalized feature vector.
Output: a predicted letter with a confidence percentage.

## Key Features

- Real-time hand detection and classification through a webcam
- Scale-invariant features, so predictions do not depend on distance from the camera
- Confidence threshold to filter out weak predictions
- Faster prediction smoothing with less delay when switching letters
- Word formation using SPACE, BACKSPACE, and C
- On-screen FPS counter
- Hand landmark and bounding box overlay
- Script to remove a mislabeled class and re-record it
- Script to migrate an older dataset to the current feature format

## Results

Accuracy depends heavily on how the data was collected, not just on the model. A dataset recorded in a single continuous take per letter will show close to 100 percent accuracy in testing regardless of how the data is split, because there is almost no variation within each class. This does not reflect real-world performance. A more meaningful accuracy check would use a separate recording session that the model has not seen during training.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| SAMPLES_PER_CLASS | 200 | Frames collected per sign during data collection |
| CONFIDENCE_THRESHOLD | 0.70 | Minimum confidence needed for a prediction to count |
| STABILITY_FRAMES | 4 | Number of consecutive matching frames before the display updates |
| N_ESTIMATORS | 100 | Number of trees in the Random Forest |
| TEST_SIZE | 0.20 | Portion of each class held out for testing (taken from the end of the recording) |

## Known Limitations and Fixes Applied

Two issues came up during live testing: predictions sometimes did not match the actual sign, and there was a noticeable delay before the correct letter appeared. A third issue was found in how the model was being evaluated.

1. Mismatched predictions. The original feature extraction only subtracted the wrist position, so features were not scale-invariant. The same sign produced different values depending on distance from the camera. Fixed by dividing all coordinates by the wrist-to-landmark-9 distance in both `collect_data.py` and `main.py`.

2. Recognition delay. The original smoothing used a majority vote over the last 10 frames, which meant a new sign needed to win out over the previous one across most of those frames before the display updated. Replaced with a simpler rule that updates the display once a prediction repeats for a few consecutive frames.

3. Evaluation accuracy was misleading. Each class was recorded in one continuous take, so nearby frames are almost identical. Any split of that same recording, random or otherwise, will still show close to 100 percent accuracy. This has not been fully solved yet. The only real fix is recording each letter across multiple separate sessions with different distances, angles, and lighting.

Other limitations that still apply:
- Works best with one hand in frame
- Poor lighting affects hand tracking
- Static gestures only; signs that involve movement (like J or Z) are not supported
- This is a demonstration and learning project, not a complete translation tool

## Future Improvements

- Collect more varied training data across multiple sessions, distances, and lighting conditions
- Try a neural network (CNN or LSTM) instead of Random Forest
- Support dynamic gestures that involve movement
- Support two-hand signs
- Add more signs, numbers, and other sign language systems
- Add text-to-speech output
- Port to a mobile app
- Add sentence-level parsing on top of individual letters
- Evaluate on a completely separate recording session rather than a split of the same one

## Hardware Requirements

- A laptop or PC with a working webcam
- At least 4 GB of RAM
- A reasonably modern CPU (Intel Core i3 or similar)
- No GPU required

## License

This project is licensed under the MIT License.

## Acknowledgements

- MediaPipe by Google, for the hand tracking framework
- Scikit-learn, for the machine learning tools
- OpenCV, for the computer vision library
