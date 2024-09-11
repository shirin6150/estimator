Project Name: Object Detector & Estimator
Description: This project involves estimating the distance of various objects in real-time using a webcam. The system utilizes camera calibration, YOLOv4 for object detection, and the triangular similarity method for distance estimation.
Table of Contents

    Installation
    Usage
    Features
    Calibration Instructions
    License
    Contact

Installation
Prerequisites

    Python 3.x
    OpenCV
    NumPy
    YOLOv4 weights and configuration files

Installation Steps

    Clone the repository:

    bash

git clone https://github.com/your-username/object-detection-distance-estimation.git

Navigate to the project directory:

bash

cd object-detection-distance-estimation

Install the required packages:

bash

    pip install -r requirements.txt

Usage
Running the Object Detection and Distance Estimation

    Capture Calibration Images

    Use the capture_image.py script to capture images of a chessboard pattern for camera calibration:

    bash

python capture_image.py

    Press s to capture and save the image.
    Press q to quit the capture process.
    Specify the directory path where the images will be saved.

Calibrate the Camera

Use the calibration.py script to calibrate the camera with the captured images:

bash

python calibration.py

    Provide the path to the images and the directory where the calibration data will be saved.
    The calibrated data will be saved in MultiMatrix_2.npz.

Object Detection and Distance Estimation

Use the DistanceEstimation.py script to perform real-time object detection and distance estimation:

bash

    python DistanceEstimation.py

Customizing the Detection

    Ensure that the classes.txt file includes all object classes you want to detect.
    Adjust object width values in the script if working with objects of different sizes.

Features

    Real-time object detection using YOLOv4.
    Distance estimation based on camera calibration and triangular similarity method.
    Video display with live distance information overlaid on detected objects.

Calibration Instructions

    Capture Images
        Use the capture_image.py file to capture images of a chessboard pattern.
        Ensure you capture approximately 60-80 images.

    Run Calibration
        Use the calibration.py file to process the images and compute calibration parameters.
        Save the calibration data as MultiMatrix_2.npz.

    Object Detection and Distance Estimation
        Use the DistanceEstimation.py script to load the calibration data and perform object detection and distance estimation.
