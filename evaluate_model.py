import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import cv2 as cv

def load_data():
    # trainingImages = np.load('training_data.npz')['arr_0']
    # trainingLabels = np.load('training_labels.npz')['arr_0']
    trainingImages = []
    trainingLabels = []
    testingImages = np.load('testing_data.npz')['arr_0']
    testingLabels = np.load('testing_labels.npz')['arr_0']
    return trainingImages, trainingLabels, testingImages, testingLabels


_, _, testingImages, testingLabels = load_data()
print('Data loaded')

testingImages = np.expand_dims(testingImages, axis = -1)
testingImages = testingImages.astype('float32') / 255.0

testingLabels_one_hot = tf.keras.utils.to_categorical(testingLabels, num_classes = 7)
print('Labels one hot encoded')

model = tf.keras.models.load_model('trained_model')
scores = model.evaluate(testingImages, testingLabels_one_hot, verbose = 1)
print('\n')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test precision:', scores[2])
print('Test recall:', scores[3])