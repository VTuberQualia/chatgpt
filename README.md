# Kickbike Readiness Analyzer

This repository provides a prototype for analyzing videos of children riding kickbikes to estimate when they are ready to transition to pedal bicycles.

## Requirements Definition

1. **Input Data**: Video footage of children riding kickbikes. Footage should capture the whole body from a side view whenever possible.
2. **Output**: An estimated readiness score or predicted timeframe for successfully riding a pedal bike.
3. **Functional Requirements**
   - Extract frames from input video and detect the child and kickbike.
   - Track body posture and motion features such as balance, steering control, and push force.
   - Train a machine learning model on labeled examples (ready vs not-ready) to learn correlations between motion features and readiness.
   - Provide an inference script that takes a new video and outputs a readiness prediction.
4. **Non‑Functional Requirements**
   - The system should run on Python 3 and rely on widely available packages (OpenCV for video processing, PyTorch for ML training).
   - Training and inference should be separated so new datasets can be added without code changes.

## Program Structure

```
kickbike_analysis/
├── __init__.py
├── data_loader.py      # Load video files and extract frames
├── feature_extractor.py# Compute movement features from frames
├── model.py            # Define ML model architecture
├── train.py            # Training entry point
└── infer.py            # Run inference on new videos
```

- **data_loader.py** handles reading video files and splitting them into frames or clips.
- **feature_extractor.py** applies computer vision techniques (e.g., pose estimation) to generate numerical features describing each frame.
- **model.py** contains a model class built with PyTorch; it may be a neural network that ingests sequences of features.
- **train.py** uses data from `data_loader` and `feature_extractor` to train the model on labeled readiness data.
- **infer.py** loads a trained model and outputs readiness scores for unseen videos.

This structure focuses on modularity and allows separate testing for each component.
