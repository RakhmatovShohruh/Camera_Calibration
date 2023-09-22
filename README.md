# **Camera Calibration with OpenCV**
![](/https://github.com/RakhmatovShohruh/info/video.gif)

## Table of Contents

1. [Overview](https://github.com/RakhmatovShohruh/Calibration#overview)
2. [Installation](https://github.com/RakhmatovShohruh/Calibration#Installation)
3. [Project Structure](https://github.com/RakhmatovShohruh/Calibration#project-structure)
4. [Execution](https://github.com/RakhmatovShohruh/Calibration#Execution)
5. [Features](https://github.com/RakhmatovShohruh/Calibration#Features)
6. [Output](https://github.com/RakhmatovShohruh/Calibration#Output)

### **Overview**

This repository contains a Python script to calibrate a camera using a set of chessboard images. 
It leverages OpenCV for image processing and calibration, as well as NumPy and Matplotlib for data management and visualization.

### **Installation**
```bash
# Clone the repository
$ git clone https://github.com/<your-github-username>/camera-calibration.git
# Navigate to the project folder.
$ cd camera-calibration
# Create the Conda environment.
$ conda env create -f environment.yml
# Activate the new environment.
$ conda activate xvision
```
### Project Structure

```
camera-calibration/
│
├── main_script.py          # Main Python script
├── images/                 # Directory for chessboard images
├── info/                   # Optional: Directory for output files
└── environment.yml         # Conda environment file
```

### Execution
1. Ensure your chessboard images for calibration are inside the images/ folder.
2. Open a terminal and navigate to the project directory.
3. Run the main script:
```bash
$ python calibration.py
```

## Features
* Automatic Chessboard Detection: Reads images from images/ and detects chessboard corners.
* Camera Calibration: Calibrates the camera using detected points.
* Parameter Persistence: Saves calibration parameters in both text and .npz formats.
* Calibration Validation: Displays a sample undistorted image and calculates the error rate of the calibration.

## Output

**info/calibration.npz:** This NumPy file contains the camera matrix, distortion coefficients, rotation vectors, and translation vectors.
