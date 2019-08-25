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
import math

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers


from utilities import absolute_file_path

def transformMatrix(theta, d, a, alpha):
    return np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)], 
                     [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)], 
                     [0, np.sin(alpha), np.cos(alpha), d], 
                     [0, 0, 0, 1]])
def transform_matrix_tensor(theta, d, a, alpha):
    matrix = [[tf.cos(theta), tf.multiply(-tf.sin(theta), tf.cos(alpha)), tf.multiply(tf.sin(theta), tf.sin(alpha)), tf.multiply(a, tf.cos(theta))], 
              [tf.sin(theta), tf.multiply(tf.cos(theta), tf.cos(alpha)), tf.multiply(-tf.cos(theta), tf.sin(alpha)), tf.multiply(a, tf.sin(theta))], 
              [tf.zeros_like(theta), tf.sin(alpha), tf.cos(alpha), d], 
              [tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)]]
    return matrix
def batch_matmul(location_v, batch_theta_v):
    zeros = tf.zeros_like(batch_theta_v)
    ones = tf.ones_like(batch_theta_v)
    m0 = transform_matrix_tensor(batch_theta_v, zeros, ones, zeros)
    m = tf.multiply(m0, location_v)
    m = tf.reduce_sum(m, axis=1)
    m = tf.transpose(m)
    
    return m
def myloss(y_true, y_pred):
    # y_true is the xy position
    # y_pred is the 2-dimensional theta output
    theta1 = y_pred[:, 0]
    theta2 = y_pred[:, 1]
    zeros = tf.zeros_like(theta1)
    zeros = K.expand_dims(zeros, axis=1)
    
    location_v = K.concatenate([zeros, zeros, zeros, zeros+1], axis=1)
    location_v = K.expand_dims(location_v, axis=-1)
    location_v = K.concatenate([location_v]*4, axis=2)
    location_v = tf.transpose(location_v, perm=[2, 1, 0])
    
    end_tip_1st_segment = batch_matmul(location_v, theta1)
    
    location_v = K.expand_dims(end_tip_1st_segment, axis=-1)
    location_v = K.concatenate([location_v]*4, axis=2)
    location_v = tf.transpose(location_v, perm=[2, 1, 0])
    
    end_tip_2nd_segment = batch_matmul(location_v, theta2)
    
    xy = end_tip_2nd_segment[:, :2]
    loss1 = K.mean(tf.maximum(K.square(xy - y_true), K.abs(xy - y_true)))
    pi = tf.constant(math.pi)
    loss2 = K.mean(tf.maximum(tf.abs(y_pred)-[[pi, 0.5 * pi]], 0))
    loss = loss1 + loss2
    return loss

# neural network sequantial model for 2 dof rootic arm
def create_model():
    model = Sequential()
    # first layer has 8 units and input_dim is dimention of input state vector
    # relu activation is simple and linear 
    ip_dim = 2 # number of input states (end effector cordinates)
    model.add(Dense(units=128, kernel_initializer="normal",activation='relu', input_dim=ip_dim))
    model.add(Dense(units=256, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=256, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=256, kernel_initializer="normal",activation='relu'))
    model.add(Dense(units=128, kernel_initializer="normal",activation='relu'))
    model.add(Dense(2, kernel_initializer="normal"))

    o = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    '''
    o = optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    o = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    o = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    o = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    o = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    o = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    o = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    '''
    # losses
    # compile model
    model.compile(loss=myloss, optimizer= o, metrics=['accuracy'])
    # show model summary
    model.summary()     
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
    print(x_train.shape)
    # load test dataset
    dataset_test = np.loadtxt(absolute_file_path('datasets/test_2dof.csv'), delimiter=",")
    x_test=dataset_test[:,:2]
    y_test=dataset_test[:,2:]
    # prediction dataset
    dataset_predict = np.loadtxt(absolute_file_path('datasets/pred_2dof.csv'), delimiter=",")
    x_predict=dataset_predict[:,:2]    

    # saves the model weights after each epoch if the validation loss decreased
    checkpointer = ModelCheckpoint(filepath=absolute_file_path('models/tmp/temp_weights.hdf5'), verbose=1, save_best_only=True)    
    # train model for loaded dataset 

    history = model.fit(x_train, y_train, batch_size=50, epochs=100, verbose=1, validation_data=(x_test, y_test),callbacks=[checkpointer])

    # Test model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Predict output
    solution = model.predict(x_predict)
    # solution=model.predict(x_predict)
    print(solution)

    # model.save(absolute_file_path('models/model_2dof.h5'))  # creates a HDF5 model file 

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