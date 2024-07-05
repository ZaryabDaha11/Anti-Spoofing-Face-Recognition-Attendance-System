import math
import time

import cv2
import cvzone
from ultralytics import YOLO

import os
import face_recognition
import pickle
import numpy as np
from datetime import datetime


confidence = 0.6

cap = cv2.VideoCapture(0)  # For Webcam use 1
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

model = YOLO("models/best.pt")

classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Finding the face encodings from the image and checking the face encodings in the image
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if conf > confidence:

                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        # print("matches", matches)
                        # print("faceDis", faceDis)

                        matchIndex = np.argmin(faceDis)
                        # print("Match Index", matchIndex)

                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        current_date = now.date()
                        year = now.year
                        month = now.month

                        if matches[matchIndex]:
                            attendance_dir = f"Attendance/{year}/{month}"
                            if not os.path.exists(attendance_dir):
                                os.makedirs(attendance_dir)

                            file_path = f"{attendance_dir}/{current_date}.csv"
                            roll_no = f'{studentIds[matchIndex]}'

                            # Check if the roll number is already in the file
                            def is_roll_no_present(file_path, roll_no):
                                if not os.path.exists(file_path):
                                    return False
                                try:
                                    with open(file_path, 'r') as f:
                                        for line in f:
                                            if line.startswith(roll_no):
                                                return True
                                    return False
                                except Exception as e:
                                    print(f"Error reading from {file_path}: {e}")
                                    return False

                            # Append the entry only if the roll number is not present
                            if not is_roll_no_present(file_path, roll_no):
                                try:
                                    with open(file_path, 'a') as f:
                                        f.write(f'{roll_no}  {current_time}\n')
                                        print(f" Attendance marked for Roll Number {roll_no}.")
                                except PermissionError:
                                    print(f"Permission denied to write to {file_path}")
                                except OSError as e:
                                    print(f"Error writing to {file_path}: {e}")
                            else:
                                print(f"Roll number {roll_no} is already present in the file.")
                            
                else:
                    color = (0, 0, 255)

                cvzone.cornerRect(img, (x1, y1, w, h),colorC=color,colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%', (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color, colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print("FPS: ",fps)

    # Background Image
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808:808+414] = imgModeList[0]

    cv2.imshow("Image", imgBackground)
    cv2.waitKey(1)