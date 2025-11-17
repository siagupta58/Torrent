from ultralytics import YOLO
import cv2
import sys

# Load your trained model
model = YOLO('military_training/military_model2/weights/best.pt')

# Get video path from command line argument
if len(sys.argv) < 2:
    print("Usage: python test_video.py <path_to_video>")
    print("Example: python test_video.py my_video.mp4")
    sys.exit(1)

video_path = sys.argv[1]

# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file: {video_path}")
    sys.exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video loaded: {video_path}")
print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
print("Press 'q' to quit, 'p' to pause/resume")
print("\nProcessing video...")

frame_count = 0
paused = False

while True:
    if not paused:
        # Read frame from video
        ret, frame = cap.read()

        if not ret:
            print("\nEnd of video reached.")
            break

        frame_count += 1

        # Run YOLO detection
        results = model(frame, conf=0.25)  # confidence threshold

        # Draw detections on frame
        annotated_frame = results[0].plot()

        # Add frame counter
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        annotated_frame = frame

    # Display the frame
    cv2.imshow('YOLO Military Detection - Q: quit, P: pause', annotated_frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Video processing complete.")
