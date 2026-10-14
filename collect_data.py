
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_PATH = "models/hand_landmarker.task"
DATA_PATH = Path("data/hand_shapes.csv")
SAMPLES_PER_CLASS = 30

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# Define the two temporary test classes.
LABELS = {
    ord("o"): "OPEN_PALM",
    ord("f"): "FIST",
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# Create CSV columns for all 21 landmarks (x, y, z).
columns = ["label"] + [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

# Preserve previously collected samples.
file_exists = DATA_PATH.exists() and DATA_PATH.stat().st_size > 0

with DATA_PATH.open("a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(columns)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        raise SystemExit

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            print("Data collection started.")
            print("O = collect OPEN_PALM sample")
            print("F = collect FIST sample")
            print("Q = quit")

            counts = {"OPEN_PALM": 0, "FIST": 0}

            # Count samples already stored for each class.
            with DATA_PATH.open(
                "r", newline="", encoding="utf-8"
            ) as existing:
                for row in csv.DictReader(existing):
                    if row["label"] in counts:
                        counts[row["label"]] += 1

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = time.monotonic_ns() // 1_000_000
                result = landmarker.detect_for_video(
                    image, timestamp
                )

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    height, width, _ = frame.shape

                    points = [
                        (int(p.x * width), int(p.y * height))
                        for p in hand
                    ]

                    for start, end in [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (17, 18), (18, 19), (19, 20),
                        (0, 17), (5, 9), (9, 13), (13, 17)
                    ]:
                        cv2.line(
                            frame, points[start], points[end],
                            (0, 255, 0), 2
                        )

                    for x, y in points:
                        cv2.circle(
                            frame, (x, y), 4, (0, 0, 255), -1
                        )

                cv2.putText(
                    frame, "O: Open palm | F: Fist | Q: Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Open palm: {counts['OPEN_PALM']}/{SAMPLES_PER_CLASS}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )

                cv2.putText(
                    frame,
                    f"Fist: {counts['FIST']}/{SAMPLES_PER_CLASS}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )

                cv2.imshow("KSL Project - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key in LABELS and result.hand_landmarks:
                    label = LABELS[key]

                    if counts[label] >= SAMPLES_PER_CLASS:
                        print(f"{label} already has enough samples.")
                        continue

                    hand = result.hand_landmarks[0]

                    # Store normalized landmark coordinates.
                    features = [
                        coordinate
                        for point in hand
                        for coordinate in (point.x, point.y, point.z)
                    ]

                    writer.writerow([label] + features)
                    file.flush()
                    counts[label] += 1

                    print(
                        f"Saved {label}: "
                        f"{counts[label]}/{SAMPLES_PER_CLASS}"
                    )

                if all(
                    count >= SAMPLES_PER_CLASS
                    for count in counts.values()
                ):
                    print("Collection target reached!")
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

print(f"Dataset saved to: {DATA_PATH}")
