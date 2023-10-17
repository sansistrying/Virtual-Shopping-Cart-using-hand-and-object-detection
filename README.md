# Virtual-Shopping-Cart-using-hand-and-object-detection
## Overview

The Hand Gesture Recognition-Based Shopping System is a modern and interactive shopping solution that uses cutting-edge technology to enhance the shopping experience for customers. This comprehensive documentation explains the project's objectives, components, functionality, and how it scales efficiently to meet the needs of large retail stores.

## How to Download and Run

Follow these steps to download and run the code:

## Clone the repository to your local machine:
   ```
   git clone https://github.com/Rohithjeevanantham/Hand-Gesture-Recognition-Based-Shopping-System.git
   ```
## Navigate to the project directory
   ```
   cd Hand-Gesture-Recognition-Based-Shopping-System
   ```
## Install the required libraries using pip
   ```
   pip install -r requirements.txt
   ```

## Run the main Python script
   ```
   python hand_gesture_shopping.py
   ```

## This project utilizes the following libraries and frameworks:

OpenCV: For object detection using YOLOv3.
Mediapipe: For hand landmark detection and tracking.
NumPy: For numerical computations.
Matplotlib: For displaying object trajectories.
Other standard Python libraries.

## Algorithms

The Hand Gesture Recognition-Based Shopping System incorporates the following algorithms and techniques:

**YOLOv3 (You Only Look Once, version 3):**
- Detects objects in real-time by dividing images into a grid and predicting bounding boxes and class probabilities.

**MediaPipe Hand Landmark Detection:**
- Tracks hand landmarks (fingertips, knuckles) for recognizing hand gestures and interactions.

**OpenCV (Open Source Computer Vision Library):**
- Handles video capture, frame processing, drawing, and user interface for visualizing results.

**Matplotlib:**
- Used to create a plot displaying the trajectory of picked-up objects.

**Action Logging:**
- Records timestamps and actions to files for later analysis of user interactions.

**Gestures Recognition:**
- Analyzes hand landmark positions to identify specific gestures, such as object pickup and placement.

These algorithms work together to provide a seamless and interactive shopping experience based on hand gestures and object recognition.


## Pre-Trained Models
   ```
   https://drive.google.com/drive/folders/1n14wF70bXs4ru2IoJGj-X8HNt6O7RNkQ
   ```
   The link to the YOLOv3 weights, YOLOv3 cfg, and COCO names files. To integrate these files into your project, you can download them and place them in the appropriate directory.

## Minimum Requirements
To run the program, ensure you have the following minimum requirements:

Python 3.x
OpenCV library installed

Mediapipe library installed

NumPy library installed

Matplotlib library installed

Webcam or camera access

## Contributors
Rohith Jeevanantham (jeevananthamrohith@gmail.com)
Rupin Ajay (rupinajay@gmail.com)
Sansita Karthikeyan (sansitakarthik2005@gmail.com)

## Contact
For any questions or support, please contact the contributors at the following email addresses provided above.
