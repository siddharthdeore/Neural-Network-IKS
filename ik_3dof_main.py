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
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adam
import keras.optimizers as Optimizers

from sklearn import preprocessing

from utilities import absolute_file_path

# neural network sequantial model
def create_model():
    model = Sequential()
    # first layer has 8 units and input_dim is dimention of input state vector
    # relu activation is simple and linear 
    model.add(Dense(units=2, kernel_initializer="normal",bias=False,activation='relu', input_dim=2))  
    model.add(Dropout(0.05))
    model.add(Dense(units=64, kernel_initializer="normal",bias=False,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(units=256, kernel_initializer="normal",bias=False,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(units=128, kernel_initializer="normal",bias=False,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(units=64, kernel_initializer="normal",bias=False,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(3, kernel_initializer="normal",bias=False))
    
    #optm = Optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #optm = Optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)
    optm = Optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #
    # compile model
    model.compile(loss='mse', optimizer=optm, metrics=['accuracy'])
    #loss mse,mean_absolute_error
    return model
    
if __name__ == '__main__':
    # use fix random seed to generate psuedorandom numbers
    np.random.seed(8)
    
    print('Creating model')
    # Model Creation
    model = create_model()

    print('Loading datasets')
    # load training dataset
    dataset_train = np.loadtxt(absolute_file_path('datasets/trains.csv'), delimiter=",")

    x_train=dataset_train[:,:2] # (input vector) first two columns 
    y_train=dataset_train[:,2:] # (output vector) second and third column
    # load test dataset
    dataset_test = np.loadtxt(absolute_file_path('datasets/tests.csv'), delimiter=",")
    x_test=dataset_test[:,:2]
    y_test=dataset_test[:,2:]

    dataset_pred = np.loadtxt(absolute_file_path('datasets/preds.csv'), delimiter=",")
    x_pred=dataset_pred[:,:2]
    y_pred=dataset_pred[:,2:]

    x_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    x_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
    y_scaler_test = preprocessing.MinMaxScaler(feature_range=(-1,1))
    x_scaler_pred = preprocessing.MinMaxScaler(feature_range=(-1,1))
    y_scaler_pred = preprocessing.MinMaxScaler(feature_range=(-1,1))
    
    dataX = x_scaler.fit_transform(x_train)
    dataY = y_scaler.fit_transform(y_train)
    dataX_test = x_scaler_test.fit_transform(x_test)
    dataY_test = y_scaler_test.fit_transform(y_test)
    dataX_pred = x_scaler_pred.fit_transform(x_pred)
    dataY_pred = y_scaler_pred.fit_transform(y_pred)
    # train model for loaded dataset 
    history = model.fit(dataX, dataY, batch_size=100, epochs=20, verbose=1, validation_data=(dataX_test, dataY_test))
    epoch=0
    err=4.0
    acc=0
    while err > 0.01 and acc < 0.9 and epoch<1000:
        epoch += 1
        err,acc = model.train_on_batch(dataX, dataY)
        print('Epoch: ' + str(epoch) + ' Training Error: ' + str(err)+ ' Training Acc: ' + str(acc))

    # Test model
    score = model.evaluate(dataX_test, dataY_test, verbose=0)
     #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])

    # Predict output
    solution=model.predict(dataX_pred)
    print(solution)
    
    #model.save('b_model_quad.h5')  # creates a HDF5 model file 

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