import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np
import cv2 as cv


datasetPath = 'dataset/'
trainingPath = [datasetPath + 'train/anger/', datasetPath + 'train/disgust/', datasetPath + 'train/fear/', datasetPath + 'train/happy/', datasetPath + 'train/neutral/', datasetPath + 'train/sad/', datasetPath + 'train/surprise/']
testingPath = [datasetPath + 'test/anger/', datasetPath + 'test/disgust/', datasetPath + 'test/fear/', datasetPath + 'test/happy/', datasetPath + 'train/neutral/', datasetPath + 'test/sad/', datasetPath + 'test/surprise/']
# trainingPath = ['dataset/train/anger/', 'dataset/train/disgust/', 'dataset/train/fear/', 'dataset/train/happy/', 'dataset/train/neutral/', 'dataset/train/sad/', 'dataset/train/surprise/']
# testingPath = ['dataset/test/anger/', 'dataset/test/disgust/', 'dataset/test/fear/', 'dataset/test/happy/', 'dataset/test/neutral/', 'dataset/test/sad/', 'dataset/test/surprise/']
emotions = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
size = 48

trainingImages = []
trainingLabels = []
testingImages = []
testingLabels = []

for folder in trainingPath:
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    for file in files:
        image = cv.imread(folder + file, cv.IMREAD_GRAYSCALE)
        trainingImages.append(image)
        trainingLabels.append(trainingPath.index(folder))
    print(folder + ' done!')
print('\n')
print('Training images loaded!')

for folder in testingPath:
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    for file in files:
        image = cv.imread(folder + file, cv.IMREAD_GRAYSCALE)
        testingImages.append(image)
        testingLabels.append(testingPath.index(folder))
    print(folder + ' done!')

np.savez('training_data.npz', np.array(trainingImages))
np.savez('training_labels.npz', np.array(trainingLabels))
np.savez('testing_data.npz', np.array(testingImages))
np.savez('testing_labels.npz', np.array(testingLabels))

print('Done!')