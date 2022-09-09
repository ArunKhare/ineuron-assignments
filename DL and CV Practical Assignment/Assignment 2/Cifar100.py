from distutils.command.build_scripts import first_line_re
from operator import mod
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
from utility import load_cifa100_data
from constant import DATA_DIR, MODEL_DIR
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam
import tensorflow as tf

data_artifact,label_artifact=load_cifa100_data(DATA_DIR)
print(data_artifact.train_data.shape)
print(data_artifact.test_data.shape)
print(label_artifact.train_coarse_labels.shape)
print(label_artifact.train_fine_labels.shape)
print(label_artifact.test_coarse_labels.shape)
print(label_artifact.test_fine_labels.shape)

# plt.imshow(data_artifact.train_data[4])
# plt.show()

model = Sequential()
model.add(Conv2D(32,kernel_size=(3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPool2D (pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(62, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Conv2D(62, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Conv2D(62, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(100, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#complile model
lrate = 0.01

opt = tf.keras.optimizers.Adam( learning_rate=lrate,name="Adam")
model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

# normalize inputs from 0-255 to 0.0 to 1.0
train_data = data_artifact.train_data.astype('float32')/255.0
test_data = data_artifact.test_data.astype('float32')/255.0
test_coarse_labels =label_artifact.test_coarse_labels
test_fine_labels = label_artifact.test_fine_labels
train_coarse_labels = label_artifact.train_coarse_labels
train_fine_labels  = label_artifact.train_fine_labels

# fit the model
model.fit(train_data,train_fine_labels)

#final evaluation of the model
scores = model.evaluate(test_data,test_fine_labels)

print(f"Accuracy {scores[1]*100}")
import os

os.makedirs('SavedModels',exist_ok=True)

model.save(MODEL_DIR)
