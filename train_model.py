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
print('Model loaded')

trainingImages, trainingLabels, _, _ = load_data()
print('Data loaded')

trainingImages = np.expand_dims(trainingImages, axis = -1)
trainingImages = trainingImages.astype('float32') / 255.0

trainingLabels_one_hot = tf.keras.utils.to_categorical(trainingLabels, num_classes = 7)
print('Labels one hot encoded')


# print('\n')
# print(trainingImages.shape)
# print(testingImages.shape)

epochs = 25
batch_size = 16

# model.fit(trainingImages, trainingLabels, epochs = epochs, batch_size = batch_size)
model.fit(trainingImages, trainingLabels_one_hot, epochs = epochs, batch_size = batch_size)
model.save('trained_model')
print('Model trained')
