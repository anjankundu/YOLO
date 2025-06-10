import cv2
from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO('yolo11n.pt')

# Open the video file
in_video = "../data/videos/dog_run.mp4"
cap = cv2.VideoCapture(in_video)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video at {in_video}")
    exit()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 inference on the frame
        results = model(frame, stream=True)

        # The results object is a generator. Loop through it.
        for r in results:
            # The .plot() method returns a BGR numpy array with detections visualized
            annotated_frame = r.plot()

            # Display the annotated frame
            cv2.imshow("Video Object Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
