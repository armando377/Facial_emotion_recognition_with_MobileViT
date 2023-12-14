import cv2 as cv
import numpy as np

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
channels = np.zeros([480, 640, 3], dtype=np.uint8)
black = np.zeros([480, 640], dtype=np.uint8)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
     
     # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # if faces is ():
    #     print("No faces found")
    
    for x, y, w, h in faces:
        h = int(h*1.05)
        cv.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
        face = cv.cvtColor(frame[y:y + h, x:x + w], cv.COLOR_BGR2GRAY)
        cv.putText(frame, 'Face detected', (x, y + h + 25), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv.imshow('Face Detection', frame)

    cv.imshow('Face Detection', frame)
    if cv.waitKey(1) == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()