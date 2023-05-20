import cv2
import face_recognition
import numpy as np


img_achille = face_recognition.load_image_file('resource/images/marie.jpg')
img_achille = cv2.cvtColor(img_achille,cv2.COLOR_RGB2BGR)
img_achi_test = face_recognition.load_image_file('resource/images/marie test.jpg')
img_achi_test = cv2.cvtColor(img_achi_test,cv2.COLOR_RGB2BGR)

face_loc = face_recognition.face_locations(img_achille)[0]
encode_achi = face_recognition.face_encodings(img_achille)[0]
cv2.rectangle(img_achille,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)

face_loc_test = face_recognition.face_locations(img_achi_test)[0]
encode_test = face_recognition.face_encodings(img_achi_test)[0]
cv2.rectangle(img_achi_test,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(255,0,255),2)

results = face_recognition.compare_faces([encode_achi],encode_test)
face_dis = face_recognition.face_distance([encode_achi],encode_test)
print(results, face_dis)
cv2.putText(img_achi_test,f'{results} {round(face_dis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Achille', img_achille)
cv2.imshow('Test achille', img_achi_test)

cv2.waitKey(0)