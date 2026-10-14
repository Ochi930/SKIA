
import cv2

for index in range(5):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if camera.isOpened():
        success, frame = camera.read()

        if success:
            print(f"Camera index {index}: WORKING")

            cv2.imshow(f"Camera index {index}", frame)
            print("Look at the image. Press any key to close it.")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Camera index {index}: opened, but no frame")

        camera.release()
    else:
        print(f"Camera index {index}: unavailable")

cv2.destroyAllWindows()
