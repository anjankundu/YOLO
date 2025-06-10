from ultralytics import YOLO
import cv2

# Load the trained YOLOv11 model
model = YOLO("yolo11n-trained.pt")

# Path to images in test directory
image_path = "../data/images/test"

# Perform inference on the images in test directory
results = model(image_path)

# Display results
for result in results:
     cv2.imshow("Object Detections", result.plot())
     cv2.waitKey(0)
