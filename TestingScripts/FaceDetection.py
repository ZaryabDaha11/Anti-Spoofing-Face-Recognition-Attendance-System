from cvzone.FaceDetectionModule import FaceDetector
import cv2
import numpy as np

# def nothing(x):
#     pass

# stating the live camera
cap = cv2.VideoCapture(0)
detector = FaceDetector()

# Check if camera opened successfully
while True:
    # Capture frame-by-frame
    success, img = cap.read()
    img, bbox = detector.findFaces(img)

    if bbox:
        #bbox - "id", "bbox", "score", "center"
        center = bbox[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        #  flipping the video frame
        # frame = cv2.flip(frame, 1)
        cv2.imshow("LiveScreen", img)
        cv2.waitKey(1)

    #     if cv2.waitKey(25) & 0xFF == ord("q"):
    #         break
    # else:
    #     break


# cap.release()
# cv2.destroyAllWindows()