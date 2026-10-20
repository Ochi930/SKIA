
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_PATH = "models/hand_landmarker.task"
DATA_PATH = Path("data/hand_shapes.csv")

# Number of NEW samples to collect for each class.
NEW_SAMPLES_PER_CLASS = 120
CAMERA_INDEX = 0
MIN_SECONDS_BETWEEN_SAMPLES = 0.5

LABELS = {
    ord("o"): "OPEN_PALM",
    ord("f"): "FIST",
}

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

columns = ["label"] + [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
    raise FileNotFoundError(
        "The original training CSV was not found. "
        "Do not create a new file; check data/hand_shapes.csv."
    )

# Verify the existing dataset before appending anything.
with DATA_PATH.open("r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    existing_header = next(reader, None)

    if existing_header != columns:
        raise ValueError(
            "Unexpected training CSV format. No samples were added."
        )

    existing_counts = {"OPEN_PALM": 0, "FIST": 0}

    for row in reader:
        if row and row[0] in existing_counts:
            existing_counts[row[0]] += 1

if any(count == 0 for count in existing_counts.values()):
    raise ValueError(
        f"Both classes must already exist. Found: {existing_counts}"
    )

new_counts = {"OPEN_PALM": 0, "FIST": 0}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
)

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError(f"Could not open webcam index {CAMERA_INDEX}.")

last_saved_time = 0.0

try:
    with DATA_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        with HandLandmarker.create_from_options(options) as landmarker:
            print("\nADDITIONAL TRAINING DATA COLLECTION")
            print("O = save an OPEN_PALM sample")
            print("F = save a FIST sample")
            print("Q = quit and keep saved progress")
            print("\nMove your hand between samples.")
            print("Vary angle, position, distance, and lighting.\n")

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp = time.monotonic_ns() // 1_000_000
                result = landmarker.detect_for_video(image, timestamp)

                hand = (
                    result.hand_landmarks[0]
                    if result.hand_landmarks
                    else None
                )

                if hand:
                    height, width, _ = frame.shape

                    points = [
                        (int(point.x * width), int(point.y * height))
                        for point in hand
                    ]

                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (17, 18), (18, 19), (19, 20),
                        (0, 17), (5, 17),
                    ]

                    for start, end in connections:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2,
                        )

                    for x, y in points:
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                messages = [
                    "O: Open palm | F: Fist | Q: Quit",
                    f"NEW Open palm: {new_counts['OPEN_PALM']}/{NEW_SAMPLES_PER_CLASS}",
                    f"NEW Fist: {new_counts['FIST']}/{NEW_SAMPLES_PER_CLASS}",
                    "Hand detected" if hand else "Show one hand to camera",
                ]

                for index, message in enumerate(messages):
                    cv2.putText(
                        frame,
                        message,
                        (10, 30 + index * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0) if index == 3 and hand else (255, 255, 255),
                        2,
                    )

                cv2.imshow("Additional Training Data", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key not in LABELS:
                    continue

                label = LABELS[key]

                if new_counts[label] >= NEW_SAMPLES_PER_CLASS:
                    print(f"New sample target reached for {label}.")
                    continue

                if hand is None:
                    print("No hand detected. Try again.")
                    continue

                now = time.monotonic()

                if now - last_saved_time < MIN_SECONDS_BETWEEN_SAMPLES:
                    continue

                features = [
                    coordinate
                    for point in hand
                    for coordinate in (point.x, point.y, point.z)
                ]

                writer.writerow([label] + features)
                file.flush()

                new_counts[label] += 1
                last_saved_time = now

                print(
                    f"New {label}: "
                    f"{new_counts[label]}/{NEW_SAMPLES_PER_CLASS}"
                )

                if all(
                    count >= NEW_SAMPLES_PER_CLASS
                    for count in new_counts.values()
                ):
                    print("\nNew training collection targets reached!")
                    break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nNew samples collected:")
for label in new_counts:
    print(f"{label}: {new_counts[label]}")

print("\nTraining dataset:", DATA_PATH)
print("Existing data has been preserved; new samples were appended.")
