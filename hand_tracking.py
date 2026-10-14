
import time
import cv2
import mediapipe as mp

# Location of the downloaded hand model
MODEL_PATH = "models/hand_landmarker.task"

# Hand skeleton connections: landmark numbers
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Little finger
    (0, 17), (5, 9), (9, 13), (13, 17)    # Palm
]

# Set up MediaPipe hand detection
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        print("Hand tracking started. Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            # Mirror the camera feed
            frame = cv2.flip(frame, 1)

            # OpenCV uses BGR; MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # VIDEO mode requires increasing timestamps
            timestamp_ms = time.monotonic_ns() // 1_000_000

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Draw landmarks for each detected hand
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    height, width, _ = frame.shape
                    points = []

                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                    # Draw connecting lines
                    for start, end in CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    # Draw joints and fingertips
                    for x, y in points:
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            -1
                        )

                message = f"Hands detected: {len(result.hand_landmarks)}"
            else:
                message = "Show your hand to the camera"

            cv2.putText(
                frame,
                message,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("KSL Interpreter - Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Hand tracking stopped.")
