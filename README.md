
# Employee Productivity & Workplace Safety Analysis

This project integrates computer vision with machine learning to analyze employee productivity and workplace safety. It combines live posture capture via a webcam with predictive modeling using both RandomForest and XGBoost algorithms.

## Overview

- **Live Posture Capture:**  
  Uses OpenCV to access the webcam, detect faces using Haar cascades, and compute a posture score (a proxy for alertness and posture quality).

- **Dataset Integration:**  
  Reads an employee dataset (`dataset.csv`) and processes it by:
  - Renaming columns (e.g., "Age" → "age", "TotalWorkingYears" → "experience", "MonthlyIncome" → "output").
  - Creating a new `productivity_level` column by binning the `output` values.
  - Simulating additional columns like `incident_severity` and `posture_score` if they are missing.

- **Predictive Modeling:**  
  Trains two sets of models:
  - **RandomForest:** For both classification (predicting productivity level) and regression (predicting incident severity).
  - **XGBoost:** For classification and regression tasks as well.
  
  The models are evaluated and used to make predictions on a sample employee record that uses the live posture score.

## Features

- **Real-Time Posture Scoring:**  
  Capture live video, detect faces, and compute posture scores using OpenCV.

- **Data Preprocessing:**  
  Automatically processes and augments the dataset with required features.

- **Ensemble Learning Models:**  
  Demonstrates proficiency with both RandomForest and XGBoost for handling classification and regression.

- **End-to-End Pipeline:**  
  Combines live data capture, dataset processing, model training, and prediction in one seamless script.

## Requirements

- Python 3.x
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

## Installation

Install the required libraries via pip:

```bash
pip install opencv-python numpy pandas scikit-learn xgboost
```

## Setup

1. Place your CSV file (named `dataset.csv`) in the same directory as the Python script (`main.py`).
2. Ensure your CSV file has the expected columns. The code renames the following:
   - `"Age"` to `age`
   - `"TotalWorkingYears"` to `experience`
   - `"MonthlyIncome"` to `output`
3. If any columns like `incident_severity` or `posture_score` are missing, they will be simulated.

## Running the Project

To run the project, execute the main Python script:

```bash
python main.py
```

### What Happens When You Run the Script:

1. **Live Posture Capture:**  
   - The webcam activates and a window opens showing the live feed.
   - The script detects your face, draws a rectangle, and overlays a computed posture score.
   - After 10 seconds (or if you press 'q'), it computes the average posture score.

2. **Dataset Loading & Processing:**  
   - The script reads `dataset.csv`, renames columns, and creates necessary additional features (e.g., productivity level).

3. **Model Training:**  
   - Two sets of models (RandomForest and XGBoost) are trained for:
     - Classification (predicting employee productivity level).
     - Regression (predicting incident severity).
   - The training performance (accuracy for classification and MSE for regression) is printed to the console.

4. **Prediction:**  
   - A new sample record is created using dummy values along with the live posture score.
   - Both model variants output predictions for productivity level and incident severity, which are printed.

## Project Structure

```
.
├── dataset.csv           # employee dataset file
├── main.py               # Main script containing the code for live capture, data processing, and modeling
└── README.md             # file
```

## Additional Notes

- **Webcam Access:**  
  Ensure you have a working webcam. The script uses OpenCV to access the default webcam.

- **Customization:**  
  You can modify the parameters (e.g., model hyperparameters, duration of video capture) as needed to better suit your data or performance needs.

- **Learning Opportunity:**  
  This project demonstrates the integration of computer vision and ensemble machine learning methods (RandomForest and XGBoost).

