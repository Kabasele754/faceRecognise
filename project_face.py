import cv2
import face_recognition
import numpy as np
import os

path = 'resource/images'
images = []
class_names = []

myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    class_names.append(os.path.splitext(cls)[0])
# print(images)
# print(class_names)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


encodeListKnow = findEncodings(images)

#print(len(encodeListKnow))
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)

        #print(faceDis)

        matchesIndex = np.argmin(faceDis)

        if matches[matchesIndex]:
            name = class_names[matchesIndex].upper()
            print(name)
            y1,y2,x2,x1 = faceLoc
            y1, y2, x2, x1 = y1*4,y2*4,x2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-31),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+5,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('Webcap', img)
    cv2.waitKey(1)