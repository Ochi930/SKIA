
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

# Configuration
MODEL_PATH = "models/hand_landmarker.task"
DATA_PATH = Path("data/hand_shapes_validation.csv")
SAMPLES_PER_CLASS = 120
CAMERA_INDEX = 0
MIN_SECONDS_BETWEEN_SAMPLES = 0.4

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

LABELS = {
    ord("o"): "OPEN_PALM",
    ord("f"): "FIST",
}

# The same 63 landmark features used by the training collector.
columns = ["label"] + [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

# Validate an existing validation file before appending to it.
file_exists = DATA_PATH.exists() and DATA_PATH.stat().st_size > 0

counts = {"OPEN_PALM": 0, "FIST": 0}

if file_exists:
    with DATA_PATH.open("r", newline="", encoding="utf-8") as existing:
        reader = csv.reader(existing)
        existing_header = next(reader, None)

        if existing_header != columns:
            raise ValueError(
                f"{DATA_PATH} has an unexpected CSV header. "
                "Back it up before making any changes."
            )

        for row in reader:
            if row and row[0] in counts:
                counts[row[0]] += 1

    if any(count > SAMPLES_PER_CLASS for count in counts.values()):
        raise ValueError(
            "The validation file already contains more samples than "
            "the configured target. Increase SAMPLES_PER_CLASS."
        )

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
    print(f"Could not open camera index {CAMERA_INDEX}.")
    raise SystemExit(1)

last_saved_time = 0.0

try:
    # Append so you can stop and resume without losing samples.
    with DATA_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(columns)
            file.flush()

        with HandLandmarker.create_from_options(options) as landmarker:
            print("\nVALIDATION DATA COLLECTION")
            print("O = save OPEN_PALM sample")
            print("F = save FIST sample")
            print("Q = quit")
            print(f"Target: {SAMPLES_PER_CLASS} samples per class")
            print("Vary position, distance, wrist angle and lighting.")
            print("Collect these samples in a separate session from training.\n")

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                # Mirror the image for a more natural webcam preview.
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

                instructions = [
                    "O: Open palm | F: Fist | Q: Quit",
                    f"Open palm: {counts['OPEN_PALM']}/{SAMPLES_PER_CLASS}",
                    f"Fist: {counts['FIST']}/{SAMPLES_PER_CLASS}",
                    "Hand detected" if hand else "Show one hand to camera",
                ]

                for index, message in enumerate(instructions):
                    cv2.putText(
                        frame,
                        message,
                        (10, 30 + index * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if index == 3 and hand else (255, 255, 255),
                        2,
                    )

                cv2.imshow("KSL Project - Validation Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key not in LABELS:
                    continue

                label = LABELS[key]

                if counts[label] >= SAMPLES_PER_CLASS:
                    print(f"{label} has reached its target.")
                    continue

                if hand is None:
                    print("No hand detected. Try again.")
                    continue

                now = time.monotonic()

                if now - last_saved_time < MIN_SECONDS_BETWEEN_SAMPLES:
                    continue

                # Preserve the original x, y, z feature order.
                features = [
                    coordinate
                    for point in hand
                    for coordinate in (point.x, point.y, point.z)
                ]

                writer.writerow([label] + features)
                file.flush()

                counts[label] += 1
                last_saved_time = now

                print(
                    f"Saved {label}: "
                    f"{counts[label]}/{SAMPLES_PER_CLASS}"
                )

                if all(
                    count >= SAMPLES_PER_CLASS
                    for count in counts.values()
                ):
                    print("\nValidation collection target reached!")
                    break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"\nValidation dataset: {DATA_PATH}")
print(f"OPEN_PALM samples: {counts['OPEN_PALM']}")
print(f"FIST samples: {counts['FIST']}")
