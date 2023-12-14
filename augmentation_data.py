import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np
import cv2 as cv


datasetPath = 'dataset/'
trainingPath = [datasetPath + 'train/anger/', datasetPath + 'train/disgust/', datasetPath + 'train/fear/', datasetPath + 'train/happy/', datasetPath + 'train/neutral', datasetPath + 'train/sad/', datasetPath + 'train/surprise/']
testingPath = [datasetPath + 'test/anger/', datasetPath + 'test/disgust/', datasetPath + 'test/fear/', datasetPath + 'test/happy/', datasetPath + 'train/neutral', datasetPath + 'test/sad/', datasetPath + 'test/surprise/']
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
        image = cv.resize(image, (size, size), interpolation = cv.INTER_AREA)
        trainingImages.append(image)
        trainingLabels.append(trainingPath.index(folder))
    print(folder + ' done!')
print('\n')
print('Training images loaded!')
# folder = 'neutral/'
# files = [f for f in listdir(folder) if isfile(join(folder, f))]
# for file in files:
#     image = cv.imread(folder + file, cv.IMREAD_GRAYSCALE)
#     image = cv.resize(image, (size, size), interpolation = cv.INTER_AREA)
#     trainingImages.append(image)
#     trainingLabels.append(4)
# print(folder + ' done!')
# print('\n')
# print('Training images loaded!')

# Directorio para guardar las imágenes aumentadas
output_dir = 'augmented_images'
os.makedirs(output_dir, exist_ok = True)

trainDataGen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 20,
    shear_range = 22,
    zoom_range = 0.1,
)
print('Image data generator created!')

trainingImages = np.expand_dims(trainingImages, axis = -1)
trainDataGen.fit(trainingImages)

# Configura el número de ejemplos generados por cada imagen original
outputs_per_example = 3
numberOfImages = [0, 0, 0, 0, 0, 0, 0]
maxImages = 4000
print('Augmenting images...')
# Genera ejemplos aumentados y guarda las imágenes
augmented_data = []
for i, (images, labels) in enumerate(trainDataGen.flow(trainingImages, trainingLabels, batch_size = outputs_per_example)):
    for j in range(outputs_per_example):
        # Guarda la imagen aumentada en una carpeta correspondiente a la clase
        output_folder = os.path.join(output_dir, f'{emotions[labels[j]]}')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'image_{i * outputs_per_example + j}.png')
        if len(listdir(output_folder)) >= maxImages:
            numberOfImages[j] = maxImages
            continue
            # break
        cv.imwrite(output_path, images[j])
        # cv.imwrite(output_path, cv.cvtColor(images[j].astype('uint8'), cv.COLOR_RGB2BGR))
print('Done!')