#!/usr/bin/env python3
"""
Sequenctial Deep Neural Network for Inverse Kinematics
"""

__author__ = "Siddharth Deore"
__version__ = "0.1.1"
__license__ = "MIT"

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

from utilities import absolute_file_path

# neural network sequantial model for 2 dof rootic arm
def create_model():
    model = Sequential()
    # first layer has 8 units and input_dim is dimention of input state vector
    # relu activation is simple and linear 
    model.add(Dense(units=8, kernel_initializer="normal",activation='relu', input_dim=2))  
    model.add(Dense(units=128, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=128, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=128, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=128, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=8, kernel_initializer="normal",activation='relu'))
    model.add(Dense(2, kernel_initializer="normal"))
    return model

    
if __name__ == '__main__':
    # use fix random seed to generate psuedorandom numbers
    np.random.seed(8)
    
    print('Creating model')
    # Model Creation
    model = create_model()

    print('Loading datasets')
    # load training dataset
    dataset_train = np.loadtxt(absolute_file_path('datasets/train_2dof.csv'), delimiter=",")
    x_train=dataset_train[:,:2] # (input vector) first two columns are end effector states
    y_train=dataset_train[:,2:] # (output vector) second and third columns are joint angles
    # load test dataset
    dataset_test = np.loadtxt(absolute_file_path('datasets/test_2dof.csv'), delimiter=",")
    x_test=dataset_test[:,:2]
    y_test=dataset_test[:,2:]
    # prediction dataset
    dataset_predict = np.loadtxt(absolute_file_path('datasets/pred_2dof.csv'), delimiter=",")
    x_predict=dataset_predict[:,:2]    
    # show model summary
    model.summary() 

    # compile model
    model.compile(loss='mean_absolute_error', optimizer= 'adam', metrics=['accuracy'])
    
    # train model for loaded dataset 

    history = model.fit(x_train, y_train, batch_size=100, epochs=100, verbose=1, validation_data=(x_test, y_test))

    # Test model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Predict output
    solution=model.predict(x_predict)
    print(solution)
    
    model.save(absolute_file_path('models/model_2dof.h5'))  # creates a HDF5 model file 

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

    del model  # deletes the existing model
