from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch
# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video
# https://www.youtube.com/watch?v=lfoTLeFooR4  Named Security_360p (Downloaded from YouTube to test the code)
cap = cv2.VideoCapture("../../Videos/Security_1080p.mp4")  # For Video

# Set the desired FPS
desired_fps = 29
cap.set(cv2.CAP_PROP_FPS, desired_fps)

model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

prev_time = time.time()

print("PyTorch Devices:", torch.cuda.device_count(), "GPU(s)" if torch.cuda.is_available() else "CPU(s)")
if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_name(0)  # Assuming you're using the first GPU (device index 0)
    print("Using GPU:", gpu_info)
else:
    print("Using CPU for computation.")

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.resize(img, (1024, 768))

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)

    # Display FPS
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to exit loop
        break

cap.release()
cv2.destroyAllWindows()


print("Process completed successfully!")
print(f"Thank you, Javier Augusto Alba Tamayo, for using this code!")
print("Just as galaxies explore the cosmic dance, your code has waltzed through data with elegance.")
print("May your AI models shine as brightly as distant stars, illuminating the path to discovery!")