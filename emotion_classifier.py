import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import cv2 as cv

path = 'dataset/train/fear/fear051.png'
# emotions = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sad', 6: 'surprise'}
emotions = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)

model = tf.keras.models.load_model('trained_model')
# model = tf.keras.models.load_model('trained_model V1')
print('Model loaded')

# image = cv.imread(path, cv.IMREAD_GRAYSCALE)
# processedImage = image.astype('float32') / 255.0
# imaprocessedImagege = np.expand_dims(processedImage, axis = -1)

face = []
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
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        h = int(h*1.05)
        cv.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
        face = cv.cvtColor(frame[y:y + h, x:x + w], cv.COLOR_BGR2GRAY)
        # cv.imshow('Face Detection', frame)
    if faces is ():
        print("No faces found")
    else:
        processedImage = cv.resize(face, (48, 48), interpolation = cv.INTER_AREA)
        processedImage = processedImage.astype('float32') / 255.0
        imaprocessedImagege = np.expand_dims(processedImage, axis = -1)
        prediction = model.predict(np.array([processedImage]), verbose = 0)
        if np.max(prediction[0]) > 0.5:
            # Guardar en text el nombre de la emoción y su probabilidad
            text = emotions[np.argmax(prediction[0])] + ': ' + str(np.max(prediction[0]))
            cv.putText(frame, text, (x, y + h + 25), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Display the resulting frame
    cv.imshow('Webcam', frame)
    if cv.waitKey(1) == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()