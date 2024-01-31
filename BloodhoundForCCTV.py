import random
import cv2
import logging
import os
from datetime import datetime, timedelta
from tkinter import Tk, Button, Label, filedialog, Toplevel, Listbox, Scrollbar, Frame, END
from tkinter import StringVar, OptionMenu, Entry
from collections import OrderedDict
from ultralytics import YOLO
import pandas as pd
import threading
import tkinter as tk
from tkinter import StringVar
import math
from tkinter import Tk, Button, Label, font, ttk  # Add 'font' to the import statement
from ttkthemes import ThemedTk


# Function to select the video file
def select_video_file():
    global video_file_path
    video_file_path = filedialog.askopenfilename()
    video_label.config(text="Selected Video: " + video_file_path)

# Function to select the output directory
def select_output_directory():
    global save_directory
    save_directory = filedialog.askdirectory()
    output_label.config(text="Output Directory: " + save_directory)

# Initialize logging
logging.basicConfig(filename='filterNsort.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Define class_list
class_list = ['black', 'blue', 'person', 'red']  # Modify this list with your actual class names

# Global variable for tracking
trackers = OrderedDict()

# Generate random colors for class list
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))]

# load a trained YOLOv8n model
model = YOLO("weights/TrainedModel.pt", "v8")

# Camera parameters
camera_focal_length = 26  # mm
image_width = 48  # width resolution
image_height = 27  # height resolution


def calculate_direction(prev_x, x):
    angle_horizontal = 2 * math.atan((x - prev_x)/(2 * camera_focal_length))
    # angle_horizontal = math.atan2(x - prev_x, camera_focal_length)

    angle_degrees_horizontal = math.degrees(angle_horizontal)

    # Adjust angles based on camera orientation (assuming the camera is facing south)
    angle_degrees_horizontal = (angle_degrees_horizontal + 0) % 360

    # Convert horizontal angle to cardinal direction
    horizontal_direction = calculate_cardinal_direction(angle_degrees_horizontal)

    return horizontal_direction

def calculate_cardinal_direction(angle_degrees):
    # Determine the number of sectors
    num_sectors = 8
    sector_size = 360 / num_sectors

    # Map the angle to the corresponding sector
    sector_index = round(angle_degrees / sector_size) % num_sectors - 1

    # Define sectors for directions
    sectors = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]

    return sectors[(sector_index + 1) % num_sectors]  # Adjusted to fix off-by-one error

# Define the function for starting object detection
def start_detection():
    global cap, out

    # Read the selected video file
    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print("Cannot open video file")
        exit()

    # Get the frame rate of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_file = save_directory + "/output.avi"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))

    # Previous centroid for direction calculation
    prev_centroids = {}

    # Get the creation time of the video file
    video_creation_time = datetime.fromtimestamp(os.path.getctime(video_file_path))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Predict on image
        detect_params = model.predict(source=[frame], conf=0.50, save=False)

        # Convert tensor array to numpy
        DP = detect_params[0].numpy()
        print(DP)

        if len(DP) != 0:
                for i in range(len(detect_params[0])):
                    print(i)

                    boxes = detect_params[0].boxes
                    box = boxes[i]  # returns one box
                    clsID = int(box.cls.numpy()[0])  # Convert clsID to integer

                    # Check if clsID is within the range of class_list
                    if class_list[clsID] == 'person':
                        conf_threshold = 0.50
                    else:
                        conf_threshold = 0.30
    
                    # Get the confidence value from the box
                    conf = box.conf.numpy()[0]

                    # Check if the detected confidence is greater than or equal to the threshold
                    if conf >= conf_threshold:
                        bb = box.xyxy.numpy()[0]

                        color = detection_colors[0]  # Use the only available color for the single class


                        cv2.rectangle(
                            frame,
                            (int(bb[0]), int(bb[1])),
                            (int(bb[2]), int(bb[3])),
                            color,
                            3,
                        )

                        # Display class name and confidence
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        if class_list[clsID] == 'person':
                            cv2.putText(
                                frame,
                                f"{class_list[clsID]} {i}: {conf*100:.2f}%", 
                                (int(bb[0]), int(bb[1]) - 40),
                                font,
                                1,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA    
                            )
                        else:
                            cv2.putText(
                                frame,
                                f"{class_list[clsID]}: {conf*100:.2f}%", 
                                (int(bb[0]), int(bb[1]) - 40),
                                font,
                                1,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA    
                            )
                    
                        # Implement centroid tracking
                        if class_list[clsID] in class_list:
                            current_class = class_list[clsID]
                            x = (bb[0] + bb[2]) / 2
                            y = (bb[1] + bb[3]) / 2
                            centroid = (int(x), int(y))

                            if clsID not in trackers:
                                trackers[clsID] = [centroid]
                                prev_centroids[clsID] = centroid
                            else:
                                trackers[clsID].append(centroid)
                                prev_x, prev_y = prev_centroids[clsID]
                                direction = calculate_direction(prev_x, x)
                                current_time = video_creation_time + timedelta(seconds=cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_rate)

                                # Display direction only for "person"
                                if current_class == 'person':
                                    detected_attributes = []
                                    for other_clsID in trackers:
                                        if other_clsID != clsID:
                                            detected_attributes.append(class_list[other_clsID])
                                    if not detected_attributes:                                        
                                        # Inside the loop where you log messages
                                        log_message = f"{current_time} - Person {i} detected - Direction: {direction}"
                                    else:
                                        # Inside the loop where you log messages
                                        log_message = f"{current_time} - Person {i} in {', '.join(detected_attributes)} detected - Direction: {direction}"

                                    logging.info(log_message)
                                    print(f"Direction: {direction}")
                                    cv2.putText(
                                        frame,
                                        f"Direction: {direction}",
                                        (int(bb[0]), int(bb[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,  # or use the font variable if defined elsewhere
                                        1,
                                        (255, 255, 255),
                                         2,
                                        cv2.LINE_AA
                                    )
                                    
                                
                                else:
                                    # For other classes (except person), don't display direction
                                    log_message = f"{current_time} - {current_class} detected"

                                    logging.info(log_message)
                                    print(f"{current_class} detected")

                                prev_centroids[clsID] = centroid
                        
                            # Draw the centroid
                            cv2.circle(frame, centroid, 4, color, -1)
                            
                            
                        else:
                            print(f"Error: clsID {clsID} out of range.")

        # Save the frame into the output video
        out.write(frame)

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    
# Define the function for starting object detection as a thread
def start_detection_thread():
    start_button.config(state="disabled")
    detection_thread = threading.Thread(target=start_detection)
    detection_thread.start()

# Create a ThemedTk window
root = ThemedTk(theme="breeze")  # You can choose a different theme

# Create a Tkinter window
root.title("Bloodhound")
root.geometry("600x500")  # Set your desired width and height
root.configure(bg="#ADD8E6")  # Light blue background color

# Improve styles using ttk themed widgets
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12, 'bold'), foreground='#ADD8E6', background='#333333')  # Dark blue button
style.configure('TLabel', font=('Helvetica', 12, 'bold'), foreground='#333333', background='#ADD8E6')  # Light blue label

# Create notebook widget
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=10)

# Create video tab
video_tab = ttk.Frame(notebook)
notebook.add(video_tab, text='Video')

# Create button for selecting video and output directory
video_button = ttk.Button(video_tab, text="Select Video File", command=select_video_file)
video_button.pack(pady=15)

video_label = ttk.Label(video_tab, text="Selected Video: ")
video_label.pack()

# Create output tab
output_tab = ttk.Frame(notebook)
notebook.add(output_tab, text='Output')

# Create button for output directory
output_button = ttk.Button(output_tab, text="Select Output Directory", command=select_output_directory)
output_button.pack(pady=15)
output_label = ttk.Label(output_tab, text="Output Directory: ")
output_label.pack()

# Create button for beginning the detection and tracking
start_button = ttk.Button(root, text="Begin", command=start_detection_thread)
start_button.pack(pady=20)

# Create a button for showing logs
logs_button = ttk.Button(root, text="Show Logs", command=lambda: show_logs(root))
logs_button.pack(pady=20)

# Define the function for showing logs with sorting and filtering options
def show_logs(root):
    log_window = Toplevel(root)
    log_window.title("Tracking Logs")
    log_window.geometry("800x600")  # Set your desired width and height for the logs window
    log_window.configure(bg="#0077cc")  

    logs = pd.read_csv('filterNsort.log', sep=' - ', names=['Time', 'Class', 'Direction'], engine='python')
    logs.sort_values(by=['Time'], inplace=True)

    # Increase the font size for logs window
    log_font = font.Font(family="Helvetica", size=12, weight="bold")

    # Define filtering function
    def filter_logs(selected_class):
        filtered_logs = logs if selected_class == "All" else logs[logs['Class'].str.contains(selected_class, case=False, regex=True)]
        update_logs(filtered_logs)

    # Define sorting function
    def sort_logs(selected_sort):
        filtered_logs = logs
        if selected_sort == "Time":
            filtered_logs.sort_values(by=['Time'], inplace=True)
        elif selected_sort == "Class":
            filtered_logs.sort_values(by=['Class'], inplace=True)
        update_logs(filtered_logs)

    # Define searching function
    def search_logs(search_query):
        filtered_logs = logs if not search_query else logs[logs.apply(lambda row: search_query.lower() in row.astype(str).str.lower().str.cat(sep=' '), axis=1)]
        update_logs(filtered_logs)

    # Define updating function
    def update_logs(filtered_logs):
        logs_listbox.delete(0, END)
        for index, row in filtered_logs.iterrows():
            logs_listbox.insert(END, f"{row['Time']} - {row['Class']} - {row['Direction']}")

    # Frame for organizing buttons
    buttons_frame = Frame(log_window, bg="#0077cc")
    buttons_frame.pack(pady=10)

    # Entry for searching
    search_var = StringVar(log_window)
    search_var.trace_add("write", lambda *args: search_logs(search_var.get()))
    search_entry = Entry(buttons_frame, textvariable=search_var, font=log_font, bg="#87ceeb", fg="#FFFFFF", width=30) 
    search_entry.grid(row=0, column=0, padx=5)

    # Dropdown for filtering
    filter_var = StringVar(log_window)
    filter_var.set("All")
    filter_dropdown = OptionMenu(buttons_frame, filter_var, "All", *class_list, command=lambda selected_class: filter_logs(selected_class))
    filter_dropdown.config(font=log_font, bg="#005fb0", fg="#bfe6f4", width=10, borderwidth=2, relief="groove", cursor='hand2')
    filter_dropdown.grid(row=0, column=1, padx=5)

    # Dropdown for sorting
    sort_var = StringVar(log_window)
    sort_var.set("Time")
    sort_dropdown = OptionMenu(buttons_frame, sort_var, "Time", "Class", command=lambda selected_sort: sort_logs(selected_sort))
    sort_dropdown.config(font=log_font, bg="#005fb0", fg="#bfe6f4", width=10, borderwidth=2, relief="groove", cursor='hand2')  
    sort_dropdown.grid(row=0, column=2, padx=5)

    # Listbox for displaying logs
    logs_listbox = Listbox(log_window, width=150, font=log_font, bg="#005fb0", fg="#FFFFFF")  
    logs_listbox.pack(side="left", fill="both", expand=True)

    # Scrollbar for the logs Listbox
    logs_scrollbar = Scrollbar(log_window, command=logs_listbox.yview)
    logs_scrollbar.pack(side="right", fill="y")

    # Configure the Listbox to use the scrollbar
    logs_listbox.config(yscrollcommand=logs_scrollbar.set)

    # Initial update of logs
    update_logs(logs)



# Start the GUI
root.mainloop()
