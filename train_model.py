import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import cv2 as cv

# emotions = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sad', 6: 'surprise'}
emotions = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def load_data():
    trainingImages = np.load('training_data.npz')['arr_0']
    trainingLabels = np.load('training_labels.npz')['arr_0']
    testingImages = np.load('testing_data.npz')['arr_0']
    testingLabels = np.load('testing_labels.npz')['arr_0']
    return trainingImages, trainingLabels, testingImages, testingLabels


model = tf.keras.models.load_model('not_trained_model')
# print('Model loaded')
# print(model.summary())

trainingImages, trainingLabels, testingImages, testingLabels = load_data()
print('Data loaded')

# index = 600 + 600 + 600 + 600 + 600 + 600 + 123
# name = emotions[testingLabels[index]]
# # name = str(trainingLabels[index])

# cv.imshow(name, testingImages[index])
# if cv.waitKey() == 27:
#     cv.destroyAllWindows()

# resized_images = [cv.resize(img, (128, 128), interpolation = cv.INTER_AREA) for img in trainingImages]
# trainingImages = resized_images
# resized_images = [cv.resize(img, (128, 128), interpolation = cv.INTER_AREA) for img in testingImages]
# testingImages = resized_images
# print('Images resized')
# cv.imshow("window", testingImages[0])
# if cv.waitKey() == 27:
#     cv.destroyAllWindows()

trainingImages = np.expand_dims(trainingImages, axis = -1)
testingImages = np.expand_dims(testingImages, axis = -1)

trainingImages = trainingImages.astype('float32') / 255.0
testingImages = testingImages.astype('float32') / 255.0

trainingLabels_one_hot = tf.keras.utils.to_categorical(trainingLabels, num_classes = 7)
testingLabels_one_hot = tf.keras.utils.to_categorical(testingLabels, num_classes = 7)
print('Labels one hot encoded')

# print(trainingLabels_one_hot[34])

print('\n')
print(trainingImages.shape)
print(testingImages.shape)

epochs = 5
batch_size = 16

# model.fit(trainingImages, trainingLabels, epochs = epochs, batch_size = batch_size)
model.fit(trainingImages, trainingLabels_one_hot, epochs = epochs, batch_size = batch_size)
model.save('trained_model')
print('Model trained')

print('\n')
# model = tf.keras.models.load_model('trained_model')
# scores = model.evaluate(testingImages, testingLabels, verbose = 1)
scores = model.evaluate(testingImages, testingLabels_one_hot, verbose = 1)
print('\n')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])