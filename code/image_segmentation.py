from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLOv11 segmentation model
model = YOLO("yolo11n-seg.pt")

# Path to images in test directory
image_path = "../data/images/test"

# Perform inference on the images in test directory
results = model(image_path)

# Process results
for result in results:
    im_array = result.plot()  # plot results, with masks and bounding boxes
    im_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR) # convert to BGR for OpenCV
    cv2.imshow("Instance Segmentation", im_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Access masks and other data
    if result.masks is not None:
        masks = result.masks.data # Mask data as tensor (num_objects, H, W)
        boxes = result.boxes.xyxy # Bounding box coordinates (num_objects, 4)
        classes = result.boxes.cls # Class IDs (num_objects)
        names = result.names # Class names mapping

        print(f"Detected {len(boxes)} objects.")
        for i in range(len(boxes)):
            class_id = int(classes[i])
            class_name = names[class_id]
            box = boxes[i].cpu().numpy().astype(int)
            mask = masks[i].cpu().numpy().astype(np.uint8) * 255 # Convert mask to 0-255

            print(f"  Object {i+1}: Class: {class_name}, BBox: {box}")

            # You can visualize individual masks if needed
            cv2.imshow(f"Semantic Segmentation: Mask for {class_name} {i+1}", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Release resources
cv2.destroyAllWindows()
