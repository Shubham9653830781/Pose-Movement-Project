# Pose Movement Project (Drone Follower Simulation)

This project uses **MediaPipe** and **OpenCV** to detect human pose landmarks from a **live webcam feed** and determine movement directions (Left, Right, Up, Down, Forward, Backward).

## Installation

Install the dependencies:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

Run the script:

```bash
python Pose_Follow.py
```

Press **Q** to quit the live feed.

## Notes
- This will **NOT run in Google Colab** because Colab does not allow direct access to your webcam.  
- Run it **locally on your PC/laptop**.
