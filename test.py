import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

test_model = keras.models.load_model('./model_ckpt')
IMG_SIZE = 28

img = cv2.imread('samples/five.png')

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape
# (28, 28)

# Resizing to a 28x28 image
# Please note my image was already in correct dimension
resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)

# 0-1 scaling
newimg = tf.keras.utils.normalize(resized, axis = 1)

# For kernal operations
newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

newimg.shape
# (1, 28, 28, 1)

predictions = test_model.predict(newimg)
num = np.argmax(predictions[0])

if int(num%2):
    label = "ODD"
else:
    label = "EVEN"

print(f"given image is {num} which is {label}")