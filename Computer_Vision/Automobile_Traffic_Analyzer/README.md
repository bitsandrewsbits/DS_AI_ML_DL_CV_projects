## **About Project**
### This program analyzes an input RGB video-stream by RTSP.
### It tries to detect cars and assigns them unique ID with detected car color.
### Also program try to detect a license plate number of detected cars.

## **Usage**
### 1. Setup separate python environment:
####    _python3 -m venv "your-env-name"_
### 2. Activate env:
####    (on linux): _source "your-env-name"/bin/activate_
####    (on windows): _"your-env-name"/bin/activate_
### 3. Install necessary libraries:
####    _pip install -r requirements.txt_
### 4. Start program:
####    _python main.py -url "URL-of-video-stream"_

#### Parameter "URL-of-video-stream" should consists from:
#####   - username = ""
#####   - password = ""
#####   - camera_ip = ""
#####   - rtsp_port(default) = 554
#####   - stream_path = ""
#### As a result video-stream-URL: "rtsp://{username}:{password}@{camera_ip}:{rtsp_port}/{stream_path}"

## **Some details about program**
### This project uses two trained YOLO models:
### 1. YOLOv8-World - model for cars recognition. Since the task was to detect only vehicles,
### this model can be configured for certain class of detection, ignoring other objects on video frames.
### 2. Fine-tuned version of YOLOv11 for License Plate Detection - model for license plate recognition.
### Program downloads it from HuggingFace via URL in global variables file.

### Also program reduces output window width(see global_vars.py) for different devices or more convinient view.

## **Program Architecture**
        RTSP Stream
            │
          Frame
            |
        YOLOv8-World (Vehicles Detection + Tracking)
            │
            |─> Vehicles IDs Cache updating
            |─> Vehicle Color Detection for new cars
            |─> Vehicle Detection Area for new cars
                        |
              License Plate Detection
               via HuggingFace model
                        |
                       OCR
                        │
            Vehicles IDs Cache updating
                        |
          Visualization of vehicles from
               Vehicles IDs Cache

## **Program Limitations**
### 1. Vehicle color estimation accomplish by average of all RGB pixels values of vechile area tensor.
### 2. License plate recognition depends on license plate area size, quality, viewing angle and lighting conditions.