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
from keras import losses

from sklearn import preprocessing
from tensoflow import sin, cos,tan

# for file handling
def absolute_file_path(rel_path):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, rel_path)

# neural network sequantial model
def create_model():
    model = Sequential()
    # first layer has 8 units and input_dim is dimention of input state vector
    # relu activation is simple and linear 
    model.add(Dense(units=2, kernel_initializer="normal",activation='relu', input_dim=2))  
    model.add(Dense(units=64, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=128, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=128, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=64, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=8, kernel_initializer="normal",activation='relu'))
    model.add(Dense(3, kernel_initializer="normal"))
    return model

    
if __name__ == '__main__':
    # use fix random seed to generate psuedorandom numbers
    np.random.seed(8)
    
    print('Creating model')
    # Model Creation
    model = create_model()

    print('Loading datasets')
    # load training dataset
    dataset_train = np.loadtxt(absolute_file_path('trains.csv'), delimiter=",")

    x_train=dataset_train[:,:2] # (input vector) first two columns 
    y_train=dataset_train[:,2:] # (output vector) second and third column
    print(x_train)
    # load test dataset
    dataset_test = np.loadtxt(absolute_file_path('tests.csv'), delimiter=",")
    x_test=dataset_train[:,:2]
    y_test=dataset_train[:,2:]
    # prediction dataset
    dataset_predict = np.loadtxt(absolute_file_path('preds.csv'), delimiter=",")
    x_predict=dataset_predict[:,:2]    
    y_predict=dataset_predict[:,2:]    
    # show model summary
    #model.summary() 

    x_scaler_train = preprocessing.MinMaxScaler(feature_range=(-1,1))
    y_scaler_train = preprocessing.MinMaxScaler(feature_range=(-1,1))
    x_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
    y_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
    x_scaler_pred = preprocessing.MinMaxScaler(feature_range=(-1,1))
    y_scaler_pred = preprocessing.MinMaxScaler(feature_range=(-1,1))

    dataX_train = x_scaler_train.fit_transform(x_train)
    dataY_train = y_scaler_train.fit_transform(y_train)
    dataX_test = x_scaler_test.fit_transform(x_test)
    dataY_test = y_scaler_test.fit_transform(y_test)
    dataX_pred = x_scaler_pred.fit_transform(x_predict)
    dataY_pred = y_scaler_pred.fit_transform(y_predict)

        

    # compile model
    model.compile(loss=losses.squared_hinge, optimizer= 'adam', metrics=['accuracy'])
    
    # train model for loaded dataset 
    history = model.fit(dataX_train, dataY_train, batch_size=20, epochs=20, verbose=1, validation_data=(dataX_test, dataY_test))

    # Test model
    score = model.evaluate(dataX_test, dataY_test, verbose=0)
     #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])

    # Predict output
    #solution=model.predict(x_predict)
    #print(solution)
    
    model.save(absolute_file_path('model_s.h5'))  # creates a HDF5 model file 

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