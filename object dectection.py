import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLOv8 model
object_detector = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

# Initialize the Deep SORT object tracker
object_tracker = DeepSort(max_age=30)

# Start capturing video from webcam (or replace 0 with 'video.mp4')
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("❌ Error: Unable to open video source.")
    exit()

print("✅ Object detection and tracking started. Press 'H' to quit.")

while True:
    frame_available, current_frame = video_capture.read()
    if not frame_available:
        break

    # Run object detection on the current frame
    detection_results = object_detector(current_frame, verbose=False)[0]

    detections_in_frame = []
    for detected_box in detection_results.boxes:
        x_min, y_min, x_max, y_max = map(int, detected_box.xyxy[0])
        confidence_score = float(detected_box.conf[0])
        class_index = int(detected_box.cls[0])
        class_label = object_detector.names[class_index]

        # Convert bounding box from x1, y1, x2, y2 to x, y, w, h
        detections_in_frame.append(
            ([x_min, y_min, x_max - x_min, y_max - y_min], confidence_score, class_label)
        )

    # Track the objects detected in the frame
    tracked_objects = object_tracker.update_tracks(detections_in_frame, frame=current_frame)

    for tracked_object in tracked_objects:
        if not tracked_object.is_confirmed() or tracked_object.track_id is None:
            continue

        track_id = tracked_object.track_id
        left, top, right, bottom = tracked_object.to_ltrb()
        x1, y1, x2, y2 = map(int, [left, top, right, bottom])
        object_label = tracked_object.get_det_class() or "object"

        # Draw bounding box and tracking ID on the frame
        cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(current_frame, f'ID {track_id}: {object_label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the processed frame
    cv2.imshow("YOLOv8 Object Detection & Tracking", current_frame)

    # Exit loop on 'H' or 'h' key press
    if cv2.waitKey(1) & 0xFF == ord('h'):
        print("🛑 Program stopped by user.")
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
