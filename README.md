# Bloodhound - Object Detection and Tracking System

Bloodhound is a Python application that utilizes YOLOv8 for object detection and tracking in video files. The application identifies specified classes, tracks their movements, and logs relevant information such as direction and timestamp. Bloodhound provides a graphical user interface (GUI) for easy configuration and interaction.

## Enhanced Analysis for 'person' Class

Bloodhound provides more effective analysis through log system filtering focused on the 'person' class. This allows users to identify clothing colors and track the motion direction of the monitored target. The filtering within the log system is tailored to enhance identification and tracking specifically for the 'person' class. Users can leverage this feature to perform more targeted and efficient analyses based on clothing color identification and tracked motion direction.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO
- Tkinter
- Pandas
- ThemedTk

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/Bloodhound.git
    cd Bloodhound
    ```

2. Install the required dependencies:

    ```bash
    pip install opencv-python
    pip install ultralytics
    pip install pandas
    pip install ttkthemes
    ```

3. Download the YOLOv8 weights file (`TrainedModel.pt`) and place it in the `weights/` directory.

## Usage

Run the application by executing the following command:
    
    ```bash
    python Bloodhound.py
    
    This will launch the GUI, allowing you to:
    
    - Click the "Select Video File" button to choose a video file for analysis.
    
    - Click the "Select Output Directory" button to specify where the processed video should be saved.
    
    - Click the "Begin" button to start the object detection and tracking process.
    
    - Click the "Show Logs" button to open a new window displaying tracking logs. Here, you can monitor detection results and tracking information.


## Configuration

Modify the `class_list` variable in the code to include the specific classes you want to detect and track. Adjust the confidence thresholds and other parameters as needed.

## Logging

Bloodhound logs tracking information to the file `filterNsort.log`. The logs include the timestamp, detected class, direction, and additional details based on the tracking logic.

## GUI Features

- **Video Tab:**
  - Select a video file for processing.
  - View the selected video file path.

- **Output Tab:**
  - Select the output directory for the processed video.
  - View the selected output directory path.

- **Main Window:**
  - Begin the object detection and tracking process.
  - Show logs with sorting and filtering options.

- **Logs Window:**
  - Search logs based on a query.
  - Filter logs based on a specific class.
  - Sort logs by time or class.

## Notes

- Ensure that the YOLOv8 weights file (`TrainedModel.pt`) is present in the `weights/` directory before running the application.

- The application will create an output video file in the specified output directory.

- The logs window provides options to filter and sort the tracking logs for better analysis.
