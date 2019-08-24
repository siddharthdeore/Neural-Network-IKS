#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import optimizers


# load datasets
dataset = np.loadtxt('ds_1900.csv', delimiter=",")
# make sure we alwayse generate psuedo-random number sequence
np.random.seed(8)
# select 15000 random rows for trainong
#idx =random.sample(range(0, dataset.shape[0]), 48000) # np.random.randint(dataset.shape[0], size=16000)
# training dataset
x_train=dataset[:1550, 3:] 
y_train=dataset[:1550, :3] 
# Delete element at index positions which are used
#dataset = np.delete(dataset, :45000, axis=0)

# select 2000 random rows for remaining dataset for testing
#idx = random.sample(range(0, dataset.shape[0]), 2000)
# testing dataset
x_test=dataset[1550:, 3:]
y_test=dataset[1550:, :3]
# Delete element at index positions which are used
#dataset = np.delete(dataset, idx, axis=0)

# select remainig rows for prediction
#x_predict=dataset[:, 3:]
#y_predict=dataset[:, :3]

print('Training dataset has ', x_train.shape[0] , 'entries of', x_train.shape[1],'input veriable(s)')
# output dataset shape
# number of collumns must equal to input dimention of first layer
print('Training dataset has ', y_train.shape[0] , 'entries of', y_train.shape[1],'output veriable(s)')



model = Sequential()
# first layer has 8 units and input_dim is dimention of input state vector
# relu activation is simple and linear 
# model.add(Dense(units=8, kernel_initializer='normal', activation='relu', input_dim=2))
#model.add(Dense(units=32, use_bias=True, kernel_initializer='normal',activation='relu'))
# model.add(Dense(units=8, kernel_initializer='normal', activation='relu', input_dim=2))
#bias_initializer='zeros', kernel_initializer='normal', kernel_initializer='random_uniform',
model.add(Dense(units=8, kernel_initializer="normal",activation='tanh', input_dim=2))  
model.add(Dense(units=64, kernel_initializer="normal",activation='tanh'))
model.add(Dense(units=128, kernel_initializer="normal",activation='tanh'))
#model.add(Dense(units=256, kernel_initializer="normal",activation='relu'))
#model.add(Dense(units=256, kernel_initializer="normal",activation='relu'))
#model.add(Dense(units=128, kernel_initializer="normal",activation='relu'))
model.add(Dense(units=64, kernel_initializer="normal",activation='tanh'))
#model.add(Dense(units=8, kernel_initializer="normal",activation='relu'))
model.add(Dense(3, kernel_initializer="normal"))

# show model summary # model.summary() 


optm = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#optm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #

model.compile(loss='mean_absolute_error', optimizer=optm, metrics=['accuracy'])  # mean squared error

# train model for loaded dataset 
history = model.fit(x_train, y_train, batch_size=5, epochs=20, verbose=1, validation_data=(x_test, y_test))

# Test model
score = model.evaluate(x_test, y_test, verbose=2)

#model.save('model_quad_1900.h5')  # creates a HDF5 model file 
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()