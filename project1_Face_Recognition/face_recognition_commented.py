# This project is about doing face and eye recognition using opencv
# Rectangle bounding boxes are used to bound each detection of face and eyes
# The input images are simply frames of real-time video from the computer camera

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 

def detect(gray, frame): #create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) 
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) #paint a rectangle around the eyes, but inside the referential of the face.
    return frame #return the image with the detected rectangles.

video_capture = cv2.VideoCapture(0) #turn the webcam on.

while True: 
    _, frame = video_capture.read() .
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    canvas = detect(gray, frame) 
    cv2.imshow('Video', canvas) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

video_capture.release() 
cv2.destroyAllWindows() 