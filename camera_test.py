
import cv2

# Open the default laptop webcam.
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access the webcam.")
    print("Check Windows camera permissions or try camera index 1.")
    raise SystemExit

print("Webcam started successfully.")
print("Move your hand in front of the camera.")
print("Press Q to close the webcam window.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read a frame from the webcam.")
        break

    # Mirror the video like a selfie camera.
    frame = cv2.flip(frame, 1)

    # Display a project label.
    cv2.putText(
        frame,
        "KSL Interpreter - Camera Test",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("KSL Interpreter", frame)

    # Exit when Q is pressed.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
