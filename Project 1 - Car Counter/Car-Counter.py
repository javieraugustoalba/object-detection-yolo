import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

class CarVelocityTracker:
    def __init__(self, video_path, yolo_weights_path, mask_path, graphics_path):
        # Initialize video capture and YOLO model
        self.video_capture = cv2.VideoCapture(video_path)
        self.object_detector = YOLO(yolo_weights_path)

        # List of class names for YOLO detection
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            # ... (other class names)
            "toothbrush"
        ]

        # Load mask image and initialize SORT tracker
        self.mask_image = cv2.imread(mask_path)
        self.object_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        # Define the active zone for velocity calculation and counting
        # The counting line is represented by two points: (start_x, y) and (end_x, y)
        # A vehicle is considered within the active zone if its center (center_x, center_y)
        # falls within the vertical range (y - vertical_tolerance, y + vertical_tolerance)
        self.counting_line_start = (300, 450)  # (start_x, y)
        self.counting_line_end = (700, 450)    # (end_x, y)
        self.vertical_tolerance = 20

        # List to keep track of detected vehicle IDs
        self.detected_vehicle_ids = []

        # Dictionary to store previous positions of vehicles
        self.previous_positions = {}

    def calculate_velocity(self, current_x, current_y, vehicle_id):
        """Calculate the velocity of a tracked vehicle based on its current and previous positions."""
        if vehicle_id in self.previous_positions:
            prev_x, prev_y = self.previous_positions[vehicle_id]
            distance = math.sqrt((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2)
            time_interval = 1  # Assuming fixed frame interval (adjust based on frame rate)
            velocity = distance / time_interval
            return velocity * 10  # Convert velocity units if necessary
        return 0

    def run(self):
        """Main loop for processing video frames and tracking vehicles."""
        while True:
            # Read a frame from the video
            success, frame = self.video_capture.read()

            # Apply mask to the frame
            masked_frame = cv2.bitwise_and(frame, self.mask_image)

            # Overlay graphics on the frame
            graphics_overlay = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED)
            frame = cvzone.overlayPNG(frame, graphics_overlay, (0, 0))

            # Perform object detection using YOLO
            detection_results = self.object_detector(masked_frame, stream=True)
            detected_objects = np.empty((0, 5))

            for result in detection_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    width, height = x2 - x1, y2 - y1
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    class_index = int(box.cls[0])
                    class_name = self.class_names[class_index]

                    if class_name in ["car", "truck", "bus", "motorbike"] and confidence > 0.3:
                        object_data = np.array([x1, y1, x2, y2, confidence])
                        detected_objects = np.vstack((detected_objects, object_data))

            # Update object tracker using SORT
            tracked_results = self.object_tracker.update(detected_objects)

            # Draw counting line
            cv2.line(frame, self.counting_line_start, self.counting_line_end, (0, 0, 255), 2)

            for tracked_object in tracked_results:
                x1, y1, x2, y2, vehicle_id = tracked_object
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width, height = x2 - x1, y2 - y1

                # Draw rectangle around the vehicle and calculate velocity
                cvzone.cornerRect(frame, (x1, y1, width, height), l=0, rt=0, colorR=(255, 0, 255))
                velocity = self.calculate_velocity(x1, y1, vehicle_id)
                cvzone.putTextRect(frame, f'Vel: {velocity:.2f} km/h', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1, offset=1)

                # Draw a circle at the center of the vehicle
                center_x, center_y = x1 + width // 2, y1 + height // 2
                cv2.circle(frame, (center_x, center_y), 1, (255, 0, 255), 1)

                # Check if the vehicle crosses the counting line
                if self.counting_line_start[0] < center_x < self.counting_line_end[0] and \
                        self.counting_line_start[1] - self.vertical_tolerance < center_y < self.counting_line_start[1] + self.vertical_tolerance:
                    if vehicle_id not in self.detected_vehicle_ids:
                        self.detected_vehicle_ids.append(vehicle_id)
                        cv2.line(frame, self.counting_line_start, self.counting_line_end, (0, 255, 0), 1)

                # Store current position for velocity calculation
                self.previous_positions[vehicle_id] = (x1, y1)

                # Print important data from the detected vehicles
                print(f"Vehicle ID: {vehicle_id}, Class: {class_name}, Confidence: {confidence}, Velocity: {velocity:.2f} km/h")

            # Display the total count of vehicles
            cv2.putText(frame, str(len(self.detected_vehicle_ids)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 1)

            # Display the processed frame
            cv2.imshow("Vehicle Velocity Tracker", frame)
            key = cv2.waitKey(1)

            # Exit loop if 'q' is pressed
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        self.video_capture.release()

        print("Process completed successfully!")
        print(f"Thank you for using this code I'm Javier Augusto Alba Tamayo please ping me +573114765457 athicadigital@gmail.com or javieraugustoalba@gmail.com!")
        print("Just as galaxies explore the cosmic dance, your code has waltzed through data with elegance.")
        print("May your AI models shine as brightly as distant stars, illuminating the path to discovery!")

if __name__ == "__main__":
    # Paths to video, YOLO weights, mask, and graphics
    video_path = "../../Videos/Traffic_3.mp4"
    yolo_weights_path = "../Yolo-Weights/yolov8l.pt"
    mask_path = "mask.png"
    graphics_path = "graphics.png"

    # Initialize and run the CarVelocityTracker
    tracker = CarVelocityTracker(video_path, yolo_weights_path, mask_path, graphics_path)
    tracker.run()
