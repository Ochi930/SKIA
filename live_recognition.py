
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import os

# Paths
MODEL_FILE = "models/hand_shape_model.pkl"
LANDMARKER_FILE = "models/hand_landmarker.task"

# Confirm required files exist
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

if not os.path.exists(LANDMARKER_FILE):
    raise FileNotFoundError(
        "Hand landmarker model not found in models/."
    )

# Load the trained classifier
saved_model = joblib.load(MODEL_FILE)
classifier = saved_model["model"]
feature_columns = saved_model["feature_columns"]

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=LANDMARKER_FILE
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open camera index 0. Check your webcam."
    )

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        print("Live recognition started.")
        print("Show an open palm or a fist.")
        print("Press Q to quit.")

        frame_timestamp_ms = 0

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a camera frame.")
                break

            # Mirror the preview for a natural webcam view
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Detect hand landmarks
            result = landmarker.detect_for_video(
                mp_image, frame_timestamp_ms
            )
            frame_timestamp_ms += 33

            prediction_text = "No hand detected"

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Draw landmarks
                height, width, _ = frame.shape

                for point in hand:
                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame, (x, y), 4, (0, 255, 0), -1
                    )

                # Prepare the 21 landmarks as 63 features
                values = []

                for point in hand:
                    values.extend([point.x, point.y, point.z])

                # Use the same feature-column order as training
                sample = pd.DataFrame(
                    [values], columns=feature_columns
                )

                prediction = classifier.predict(sample)[0]
                prediction_text = str(prediction)

            # Display prediction
            cv2.rectangle(
                frame, (10, 10), (frame.shape[1] - 10, 65),
                (30, 30, 30), -1
            )

            cv2.putText(
                frame,
                prediction_text,
                (25, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.imshow("KSL Interpreter - Live Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Live recognition stopped.")
