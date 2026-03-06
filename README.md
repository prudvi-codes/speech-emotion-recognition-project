# Speech Emotion Recognition using Machine Learning

## Project Overview

This project implements a Speech Emotion Recognition (SER) system using Machine Learning. The system analyzes speech audio and detects the emotional state of a speaker using acoustic features extracted from the audio signal.

The project uses **MFCC (Mel Frequency Cepstral Coefficients)** for feature extraction and a **Support Vector Machine (SVM)** classifier for emotion classification.

---

## Dataset

The model is trained using the **RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song)**.

Dataset features:

* 24 professional actors
* 1440 speech recordings
* WAV audio format
* Multiple emotional expressions

Dataset Link:
https://zenodo.org/record/1188976

---

## Source Code Files

### 1. feature_extraction.py

This file contains the function used to extract audio features from speech signals.

Main tasks performed:

* Load audio files
* Extract MFCC features
* Convert audio signals into numerical feature vectors

These features are later used to train the machine learning model.

---

### 2. train_model.py

This file is responsible for training the emotion recognition model.

Main tasks performed:

* Load dataset audio files
* Extract MFCC features using the feature extraction module
* Split dataset into training and testing sets
* Train a Support Vector Machine (SVM) classifier
* Evaluate model accuracy
* Save the trained model as `emotion_model.pkl`

---

### 3. predict_emotion.py

This file is used to predict emotions from a new audio file.

Main tasks performed:

* Load the trained machine learning model
* Extract MFCC features from a test audio file
* Predict the emotion using the trained classifier
* Display the predicted emotion

---

### 4. requirements.txt

This file contains the Python libraries required to run the project.

Required libraries include:

* librosa
* numpy
* scikit-learn
* soundfile

These libraries can be installed using the command:

```bash
pip install -r requirements.txt
```

---

## Project Workflow

Speech Input
↓
Feature Extraction (MFCC)
↓
Machine Learning Model Training
↓
Emotion Classification

---

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train_model.py
```

### Step 3: Predict Emotion

```bash
python predict_emotion.py
```

---

## Author

Prudviraj
B.Tech – Electronics and Communication Engineering
