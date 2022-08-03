import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import tensorflow_addons as tfa

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("=== library import done ===")

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

print("=== data loaded ===")

print("=== rotating the data ===")
rotated_x_train = []
for img in tqdm(x_train):
  rotate = tfa.image.rotate(img, tf.constant(180.0))
  rotated_x_train.append(rotate)

rotated_x_test = []
for img in tqdm(x_test):
  rotate = tfa.image.rotate(img, tf.constant(180.0))
  rotated_x_test.append(rotate)

rotated_x_train_final = np.asarray(rotated_x_train)
rotated_x_test_final = np.asarray(rotated_x_test)

x_train_final = np.concatenate((rotated_x_train_final, x_train))
x_test_final = np.concatenate((rotated_x_test_final, x_test))

y_train_final = np.concatenate((y_train, y_train))
y_test_final = np.concatenate((y_test, y_test))

print("=== data rotation done, training and test set updated, with labels as well ===")

# # Make odd-1/even-0 labels

y_train_NUM = np.asarray([int(x%2) for x in y_train_final])
y_test_NUM = np.asarray([int(x%2) for x in y_test_final])


# Normalization
print("=== normalizing the data ===")
x_train_final = tf.keras.utils.normalize(x_train_final, axis = 1)
x_test_final = tf.keras.utils.normalize(x_test_final, axis = 1)

IMG_SIZE=28
# -1 is a shorthand, which returns the length of the dataset
x_trainr = np.array(x_train_final).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr = np.array(x_test_final).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Training Samples dimension", x_trainr.shape)
print("Testing Samples dimension", x_testr.shape)


print("=== model init... ===")
# Creating the network
model = Sequential()

### First Convolution Layer
# 64 -> number of filters, (3,3) -> size of each kernal,
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:])) # For first layer we have to mention the size of input
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Second Convolution Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Third Convolution Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Fully connected layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

### Fully connected layer 2
model.add(Dense(32))
model.add(Activation("relu"))

### Fully connected layer 3, output layer must be equal to number of classes
model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_trainr, y_train_final, epochs=10, validation_split = 0.3)

# Evaluating the accuracy on the test data
test_loss, test_acc = model.evaluate(x_testr, y_test_final)
print("=== Test Loss on 10,000 test samples ===", test_loss)
print("=== Test Accuracy on 10,000 test samples ===", test_acc)

print("=== ODD/EVEN prediction eval ===")
predictions = model.predict([x_testr])
test_preds = np.argmax(predictions, axis=1)

test_preds_odd_even = np.asarray([int(x%2) for x in test_preds])

print("=== odd even test accuracy ===", np.mean(test_preds_odd_even==y_test_NUM))

## Save model
model.save('./model_ckpt')
print('=== model saved in model_ckpt dir ===')