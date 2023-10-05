# Import necessary libraries
from ultralytics import YOLO  # Import YOLO object detection model from ultralytics
import cv2  # Import OpenCV for image processing
import cvzone  # Import cvzone for additional computer vision utilities
import math  # Import math for mathematical operations
import time  # Import time to calculate FPS
import torch  # Import torch to check for GPU availability

# Uncomment below lines to use webcam as input
# cap = cv2.VideoCapture(0)  # Initialize webcam. '0' refers to default webcam
# cap.set(3, 1280)  # Set the width of the frames
# cap.set(4, 720)   # Set the height of the frames

# Uncomment below to use video file as input
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # Load a video of bikes
cap = cv2.VideoCapture("../../Videos/2.mp4")  # Load a musical video with people in dark light
cap.set(cv2.CAP_PROP_FPS, 29)  # Set the FPS of the video capture object to 29

# Load the YOLO model with specified weights
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Define class names that the YOLO model can detect
# These names are used to interpret the model's output
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  # List of class names

# Variables to calculate Frames Per Second (FPS)
prev_frame_time = 0  # Previous frame time
new_frame_time = 0  # Current/new frame time

# Desired FPS and time per frame (in seconds)
desired_fps = 29  # Desired Frames Per Second
frame_time = 1 / desired_fps  # Time per frame according to desired FPS

# Check and display computation device (GPU/CPU) information
print("PyTorch Devices:", torch.cuda.device_count(), "GPU(s)" if torch.cuda.is_available() else "CPU(s)")
if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_name(0)  # Get name of the GPU device
    print("Using GPU:", gpu_info)
else:
    print("Using CPU for computation.")

# Main loop to process each frame from the webcam/video
while True:
    new_frame_time = time.time()  # Record the time at the start of the loop

    success, img = cap.read()  # Read a frame from the video capture object
    results = model(img, stream=True)  # Run the YOLO model on the frame

    # Iterate through the results
    for r in results:
        boxes = r.boxes  # Get bounding boxes

        # Iterate through each bounding box
        for box in boxes:
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            w, h = x2 - x1, y2 - y1  # Width and height of the box
            cvzone.cornerRect(img, (x1, y1, w, h))  # Draw rectangle using cvzone

            # Extract confidence score and class index
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence of detection
            cls = int(box.cls[0])  # Class index

            # Add label and confidence score to the bounding box
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)  # Calculate FPS
    prev_frame_time = new_frame_time  # Update previous frame time
    print(fps)  # Print FPS in the console

    # Display FPS on the frame
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Image", img)

    # Break the loop if 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value of the 'Esc' key
        break
