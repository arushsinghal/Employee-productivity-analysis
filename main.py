import cv2
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# -------------------------
# 1. Live Posture Capture using OpenCV
# -------------------------
def capture_posture_score(duration=10):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    start_time = time.time()
    scores = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_center_y = y + h / 2
            frame_height = frame.shape[0]
            score = face_center_y / frame_height
            scores.append(score)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Score: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Live Posture Capture (Press 'q' to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > duration:
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.mean(scores) if scores else None

# -------------------------
# 2. Load and Integrate the Dataset
# -------------------------
def load_dataset(csv_file='dataset.csv'):
    data = pd.read_csv(csv_file)
    # Rename columns to expected names (adjust these if your CSV uses different names)
    data.rename(columns={'Age': 'age', 'TotalWorkingYears': 'experience', 'MonthlyIncome': 'output'}, inplace=True)
    quantiles = data['output'].quantile([0.33, 0.66]).values
    def assign_productivity(x):
        if x <= quantiles[0]:
            return "Low"
        elif x <= quantiles[1]:
            return "Medium"
        else:
            return "High"
    data['productivity_level'] = data['output'].apply(assign_productivity)
    if 'incident_severity' not in data.columns:
        data['incident_severity'] = np.random.uniform(0, 10, size=len(data))
    if 'posture_score' not in data.columns:
        data['posture_score'] = np.random.uniform(0.3, 0.7, size=len(data))
    return data

# -------------------------
# 3. Train Models (RandomForest Variants)
# -------------------------
def train_classification_model_rf(data):
    features = data[['age', 'experience', 'output', 'posture_score']]
    target = data['productivity_level']
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"RandomForest Classification Model Accuracy: {acc:.2f}")
    return model_rf, le

def train_regression_model_rf(data):
    features = data[['age', 'experience', 'output', 'posture_score']]
    target = data['incident_severity']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model_rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf_reg.fit(X_train, y_train)
    y_pred = model_rf_reg.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"RandomForest Regression Model MSE: {mse:.2f}")
    return model_rf_reg

# -------------------------
# 4. Train Models (XGBoost Variants)
# -------------------------
def train_classification_model_xgb(data):
    features = data[['age', 'experience', 'output', 'posture_score']]
    target = data['productivity_level']
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    model_xgb = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100,
                              use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Classification Model Accuracy: {acc:.2f}")
    return model_xgb, le

def train_regression_model_xgb(data):
    features = data[['age', 'experience', 'output', 'posture_score']]
    target = data['incident_severity']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model_xgb_reg = XGBRegressor(objective='reg:squarederror', max_depth=5, learning_rate=0.1,
                                 n_estimators=100, random_state=42)
    model_xgb_reg.fit(X_train, y_train)
    y_pred = model_xgb_reg.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"XGBoost Regression Model MSE: {mse:.2f}")
    return model_xgb_reg

# -------------------------
# 5. Main Execution
# -------------------------
def main():
    print("Starting live posture capture. Press 'q' to exit early or wait for the duration to complete.")
    live_posture_score = capture_posture_score(duration=10)
    if live_posture_score is None:
        print("No posture score captured. Exiting.")
        return
    print(f"Average Live Posture Score: {live_posture_score:.2f}")
    
    try:
        data = load_dataset('dataset.csv')
    except Exception as e:
        print("Error loading dataset:", e)
        return
    
    # Train RandomForest models
    rf_clf, rf_le = train_classification_model_rf(data)
    rf_reg = train_regression_model_rf(data)
    
    # Train XGBoost models
    xgb_clf, xgb_le = train_classification_model_xgb(data)
    xgb_reg = train_regression_model_xgb(data)
    
    # Create a new sample using the live posture score and dummy values for other features
    new_sample = pd.DataFrame({
        'age': [30],
        'experience': [5],
        'output': [80],
        'posture_score': [live_posture_score]
    })
    
    # RandomForest predictions
    pred_rf_class = rf_clf.predict(new_sample)
    pred_rf_class_label = rf_le.inverse_transform(pred_rf_class)
    print("RandomForest Predicted Productivity Level for the live sample:", pred_rf_class_label[0])
    pred_rf_reg = rf_reg.predict(new_sample)
    print("RandomForest Predicted Incident Severity for the live sample:", pred_rf_reg[0])
    
    # XGBoost predictions
    pred_xgb_class = xgb_clf.predict(new_sample)
    pred_xgb_class_label = xgb_le.inverse_transform(pred_xgb_class)
    print("XGBoost Predicted Productivity Level for the live sample:", pred_xgb_class_label[0])
    pred_xgb_reg = xgb_reg.predict(new_sample)
    print("XGBoost Predicted Incident Severity for the live sample:", pred_xgb_reg[0])

if __name__ == '__main__':
    main()
