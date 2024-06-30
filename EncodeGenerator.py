import cv2
import face_recognition
import pickle
import os

# Importing the images
folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
studentIds = []
for path in pathList:
    img = cv2.imread(os.path.join(folderPath, path))
    if img is None:
        print(f"Error reading image: {path}")
        continue
    imgList.append(img)
    studentIds.append(os.path.splitext(path)[0])


    # print(os.path.splitext(path)[0])
# print(studentIds)


def findEncodings(imgList):
    encodeList = []
    for img in imgList:
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)
            if len(encodings) > 0:
                encodeList.append(encodings[0])
            else:
                print(f"No faces found in image: {img.shape}")
        except Exception as e:
            print(f"Error processing image: {e}")
    return encodeList

print('Encoding Started...')
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
# print('Encoding: ', encodeListKnown)
print('Encoding Completed')

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")