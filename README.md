# SafeTrail – AI-Powered Human Behaviour Analysis and Anomaly Detection

## Overview

**SafeTrail** is a computer vision-based human behaviour analysis system designed to identify potentially abnormal human activities from video footage.

The system combines **YOLOv11 Pose Estimation, custom YOLO-based frame classification, ByteTrack object tracking, and temporal pose analysis** to detect and monitor human activities over time.

Unlike approaches that analyse individual frames independently, SafeTrail maintains person-specific movement histories to understand changes in body posture, movement patterns, and activity over multiple frames.

The system can identify behaviours such as:

- Fighting
- Running
- Walking
- Standing
- Fainting

It also generates annotated video output containing human skeletons, tracking IDs, behaviour labels, and anomaly statistics.

## Key Features

- **Human Pose Estimation:** Uses YOLOv11x-Pose to extract 17 human body keypoints.
- **Multi-Object Tracking:** Uses ByteTrack to maintain consistent person IDs across video frames.
- **Temporal Behaviour Analysis:** Analyses pose histories, movement patterns, velocity, and acceleration.
- **Abnormal Behaviour Detection:** Identifies potentially abnormal activities using movement and posture-based indicators.
- **Frame-Level Classification:** Uses a separate custom-trained YOLO model to classify video frames.
- **Video Visualization:** Displays skeletons, tracking IDs, behaviour labels, and confidence information.
- **Anomaly Statistics:** Calculates the percentage of frames classified as abnormal.

## System Architecture

```text
              Input CCTV / Video
                      |
                      v
              Video Frame Extraction
                      |
                      v
             Image Preprocessing
                      |
          +-----------+-----------+
          |                       |
          v                       v
   YOLOv11x-Pose           Custom YOLO Model
   Pose Estimation         Frame Classification
          |                       |
          v                       v
   Human Keypoints          Frame-Level
   and Detections            Predictions
          |
          v
   ByteTrack Tracking
          |
          v
   Person ID Assignment
          |
          v
   EnhancedPoseAnalyzer
          |
          v
   Temporal Feature Extraction
          |
          v
   Behaviour Classification
          |
          v
   Anomaly Analysis and
   Behaviour Aggregation
          |
          v
   Annotated Video Output
   + Anomaly Statistics
```

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Object Detection | Ultralytics YOLO |
| Pose Estimation | YOLOv11x-Pose |
| Object Tracking | ByteTrack |
| Video Processing | OpenCV |
| Visualization | Supervision, OpenCV |
| Data Structures | Deque, Dictionary, DefaultDict |

## System Components

### 1. Video Input and Frame Extraction

The system accepts video footage as input and extracts individual frames for processing.

OpenCV and the Supervision library are used for video processing and frame extraction.

### 2. Image Preprocessing

Preprocessing techniques explored in the implementation include:

- Gaussian blur for background smoothing.
- Gamma correction for brightness adjustment.
- Denoising to reduce unwanted image noise.
- Sharpening to improve visual details.
- YOLO segmentation-based person masking in a preprocessing branch.

These techniques are intended to improve visual quality under different video conditions.

### 3. Human Pose Estimation

The system uses the pretrained **YOLOv11x-Pose** model to detect people and estimate their body keypoints.

It extracts 17 human keypoints based on the COCO human pose format, including:

- Nose
- Eyes and ears
- Shoulders
- Elbows
- Wrists
- Hips
- Knees
- Ankles

The keypoints provide information about body posture and limb positions, which is used in subsequent behaviour analysis.

### 4. Multi-Person Tracking Using ByteTrack

ByteTrack associates detected people across consecutive video frames.

Each tracked person receives a consistent tracking ID, allowing the system to maintain separate movement histories for different individuals.

This is important because behaviour analysis requires understanding how a particular person's posture and movement change over time.

### 5. EnhancedPoseAnalyzer – Temporal Analysis

The `EnhancedPoseAnalyzer` module is responsible for analysing person-specific pose information over multiple frames.

It maintains historical information using data structures such as dictionaries and deques.

The features considered include:

- Pose history
- Keypoint movement
- Velocity history
- Acceleration history
- Body-centre movement
- Cumulative movement distance
- Direction changes
- Limb coordination
- Behaviour consistency

The implementation uses a temporal buffer to retain recent observations and analyse movement patterns rather than relying on a single frame.

### 6. Behaviour Classification

The system uses pose-based movement indicators to identify different activities.

| Behaviour | Analysis |
|---|---|
| Fighting | Analyses arm and body movements, movement variation, and stance-related indicators. |
| Running | Uses movement magnitude and temporal movement patterns. |
| Walking | Identifies comparatively regular body movement. |
| Standing | Identifies low movement over time. |
| Fainting | Analyses changes in head and hip positions. |

The implementation also applies temporal consistency checks to reduce unstable behaviour predictions.

### 7. Frame-Level Classification

A separate finetuned YOLO model (`best.pt`) is used for frame-level classification.

This model provides an additional signal about whether a frame is classified as normal or abnormal.

It complements the person-level pose analysis; it is not a jointly trained fusion model.

### 8. Visualization and Anomaly Statistics

The final output includes:

- Human skeleton visualization.
- Person tracking IDs.
- Behaviour labels.
- Prediction confidence.
- Frame-level anomaly information.
- Overall anomaly percentage.

The frame-level anomaly rate is calculated as:

\[
\text{Anomaly Rate} =
\frac{\text{Number of Abnormal Frames}}
{\text{Total Processed Frames}}\times100
\]

This represents the proportion of frames classified as abnormal, not the percentage of people exhibiting abnormal behaviour.

## Project Workflow

1. Load the input video.
2. Extract video frames.
3. Apply the selected preprocessing operations.
4. Run YOLOv11x-Pose to detect people and estimate body keypoints.
5. Apply ByteTrack to associate detections across frames.
6. Maintain individual pose histories using tracking IDs.
7. Extract temporal movement features.
8. Analyse posture and movement to identify behaviours.
9. Run the separate custom YOLO frame classifier.
10. Generate annotated video output and anomaly statistics.

## Model Information

| Model / Module | Responsibility |
|---|---|
| YOLOv11x-Pose | Human detection and 17-keypoint pose estimation |
| Custom YOLO (`best.pt`) | Frame-level classification |
| ByteTrack | Multi-person tracking |
| EnhancedPoseAnalyzer | Temporal movement and pose analysis |

The pose estimation model and custom frame classifier serve different purposes within the pipeline.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python environment.
- PyTorch.
- Ultralytics.
- OpenCV.
- Supervision.
- NumPy.

### Install Dependencies

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install supervision
pip install numpy
```

### Model Weights

Ensure that the required model weights are available:

```text
yolo11x-pose.pt
best.pt
```

Update the model paths in the notebook to match their locations on your system.

### Running the Project

1. Clone or download the project repository.
2. Install the required dependencies.
3. Download or provide the required model weights.
4. Configure the input video path.
5. Configure the model weight paths.
6. Execute the notebook cells in order.
7. Inspect the generated annotated video and anomaly statistics.

## Expected Output

The system generates an annotated video showing:

- Detected individuals.
- Human pose skeletons.
- Person tracking IDs.
- Predicted behaviour labels.
- Confidence scores.
- Frame-level anomaly information.

It also calculates an aggregate anomaly rate for the processed video.

## Applications

- Intelligent CCTV surveillance.
- Public-space safety monitoring.
- Suspicious activity analysis.
- Human behaviour monitoring.
- Automated video analysis.
- Research in computer vision and human activity recognition.

## Limitations and Future Improvements

- **Lighting conditions:** Poor lighting and occlusion can affect pose estimation accuracy.
- **Crowded environments:** Overlapping people can make tracking and pose estimation difficult.
- **Behaviour ambiguity:** Similar movements may correspond to different activities.
- **Threshold sensitivity:** Rule-based temporal behaviour indicators may require tuning for different environments.
- **Real-time performance:** Processing speed depends on video resolution, model size, and available GPU resources.

### Future Enhancements

- Integrate stronger temporal deep learning models for behaviour recognition.
- Improve tracking consistency in crowded environments.
- Evaluate performance across additional surveillance datasets.
- Optimize inference speed for real-time CCTV deployment.
- Improve robustness to camera angle, lighting, and partial occlusion.

## Research Context

This project was developed in the context of computer vision-based human behaviour analysis, with research experience involving surveillance datasets such as CHAD and UCF-Crime.

## Authors

Developed with love and teamwork as part of a research internship project focused on AI-powered human behaviour analysis and anomaly detection.

**By:** Kavana H, Anshul RS & Sharvari V Ram


