import cv2
import face_recognition
# 1, prepare people's face
 # #    
me=cv2.imread('test.jpg')
him=cv2.imread('lena.jpg')

 # Code the picture
me_face_encoding=face_recognition.face_encodings(me)[0]
him_face_encoding=face_recognition.face_encodings(him)[0]

 # Prepare a list of face codes in human face
known_face_encodings=[me_face_encoding,him_face_encoding]
 # Prepare people's face coding corresponding name
known_face_names=['me','him']

 # 2, capture pictures in video
vc=cv2.VideoCapture(0)
while True:
    ret,img=vc.read()
    if not ret:
        print('does not capture video')
        break
         # 3, discover the position of the face in the picture
    locations=face_recognition.face_locations(img)
         #    
    face_encodings=face_recognition.face_encodings(img,locations)
         # Traverage Locations, Face_Encodings, identify people in the picture
    for (top,right,bottom,left), face_encoding in zip(locations,face_encodings):
                 # 4, identify the name of the face in the picture in the video
        matchs=face_recognition.compare_faces(known_face_encodings,face_encoding)
        name='unknown'
        for match,known_name in zip(matchs,known_face_names):
            if match:
                name=known_name
                break
                 #       position
        cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
                 #       name
        cv2.putText(img,name,(left,top-20),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
                 # 5, show
        cv2.imshow('Video',img)
                 # 6, release
        if cv2.waitKey(1) !=-1:
            vc.release()
            cv2.destroyAllWindows()
            break