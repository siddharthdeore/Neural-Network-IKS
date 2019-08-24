import keras 
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LeakyReLU, Input, Lambda, Concatenate
from keras.losses import mean_absolute_error, mean_squared_error
import os
import sys

import matplotlib.pyplot as plt
from keras import optimizers
from keras import metrics
import scipy.io
import tensorflow as tf
import keras.backend as K
from IPython.display import clear_output
import math

def transform_matrix(theta, d, a, alpha):
    return np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)], 
                     [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)], 
                     [0, np.sin(alpha), np.cos(alpha), d], 
                     [0, 0, 0, 1]])
  
def forward_kinematics_2(theta1, theta2):
    T00 = transform_matrix(theta1,0,1,0)
    T01 = transform_matrix(theta2,0,1,0)
    pos = [0, 0, 0, 1]
    Etip = np.matmul(np.matmul(T00, T01), pos)
    return T00, T01, Etip

def get_positions_2(theta):
    # assuming theta is already in radian
    theta1 = theta[0]
    theta2 = theta[1]
    
    T00, T01, Etip = forward_kinematics_2(theta1, theta2)
    t = np.transpose(np.array([[0, 0, 0, 1]]))
    pos_1 = np.matmul(T00, t)
    
    # only return first 2 elements as xy
    return np.array([pos_1[:2], np.reshape(Etip[:2], (2, 1))])
  
def transform_matrix_tensor(theta, d, a, alpha):
    # tensor version of transform matrix
    matrix = [[tf.cos(theta), tf.multiply(-tf.sin(theta), tf.cos(alpha)), tf.multiply(tf.sin(theta), tf.sin(alpha)), tf.multiply(a, tf.cos(theta))], 
              [tf.sin(theta), tf.multiply(tf.cos(theta), tf.cos(alpha)), tf.multiply(-tf.cos(theta), tf.sin(alpha)), tf.multiply(a, tf.sin(theta))], 
              [tf.zeros_like(theta), tf.sin(alpha), tf.cos(alpha), d], 
              [tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)]]
    return matrix
  
def batch_matmul(location_v, batch_theta_v):
    # perform matrix multiplication between the location vector and the transform matrix, 
    # independently for each example in the batch, but done in a parallel way
    zeros = tf.zeros_like(batch_theta_v)
    ones = tf.ones_like(batch_theta_v)
    m0 = transform_matrix_tensor(batch_theta_v, zeros, ones, zeros)
    m = tf.multiply(m0, location_v)
    m = tf.reduce_sum(m, axis=1)
    m = tf.transpose(m)
    return m
  
def forward_kinematics_loss_2(y_true, y_pred):
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
    loss1 = K.mean(K.square(xy - y_true))
    pi = tf.constant(math.pi)
    loss2 = K.mean(tf.maximum(tf.abs(y_pred)-[[pi, 0.5 * pi]], 0))
    loss = loss1 + loss2
    return loss
  
def get_xy_and_theta_2(num):
    xy = np.zeros((num, 2))
    theta = np.zeros((num, 3))

    theta[:,0] = (np.random.random((num)) * 2 * np.pi) - np.pi
    theta[:,1] = (np.random.random((num)) * np.pi) - (0.5 * np.pi)
    for i in range(num):
        _, _, temp = forward_kinematics_2(theta[i,0], theta[i,1])
        xy[i, :] = temp[:2]
    return xy, theta

K.clear_session()

model = Sequential([
    Dense(256, input_shape=(2,)),
    LeakyReLU(),
    Dense(256),
    LeakyReLU(),
    Dense(256),
    LeakyReLU(),
    Dense(256),
    LeakyReLU(),
    Dense(256),
    LeakyReLU(),
    Dense(256),
    LeakyReLU(),
    Dense(256),
    LeakyReLU(),
    Dense(2)       # <==== Change this to the number of angles predicted
])

adam = optimizers.Adam(lr=1e-6)
model.compile(optimizer=adam,
              loss=forward_kinematics_loss_2)

loss_hist = []
error_hist = []

EPOCHS = 100000
xy_test, theta_test = get_xy_and_theta_2(10000)

for i in range(EPOCHS):
    # train on a mini-batch
    print("epoch {}".format(i))
    xy_train, theta_train = get_xy_and_theta_2(100)
    history = model.fit(xy_train, xy_train, epochs=1, batch_size=1, verbose = 1)
    
    # test the model on the test set
    theta_pred = model.predict(xy_test)
    xy_pred = np.zeros((theta_pred.shape[0], 2))
    for j in range(theta_pred.shape[0]):
        a = get_positions_2(np.squeeze(theta_pred[j, :]))
        xy_pred[j, :] = a[1, :, 0]
    error = np.mean(np.square(xy_pred - xy_test))
    
    # plot (1) loss & (2) mean square error on test set, vs. training steps
    loss_hist.append(history.history['loss'][0])
    error_hist.append(error)
    clear_output()
    plt.figure(figsize=(16, 4))
    line1, = plt.plot(error_hist, label="error hist")
    line2, = plt.plot(loss_hist, label="loss hist")
    plt.grid()
    plt.title('mean squraed error on test set vs. epoch')
    plt.legend((line1, line2), ('error hist', 'loss hist'))
    plt.show()

    # randomly showcase 12 examples to visually see how the network is doing
    xy_temp, theta_temp = get_xy_and_theta_2(12)
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            idx = j + i * 4
            theta = model.predict(np.reshape(xy_temp[idx], (1, 2)))
            
            # plot xy from predicted angles and ground truth, for 2-segment arm
            a = get_positions_2(np.squeeze(theta))
            col.plot([0, a[0][0]], [0, a[0][1]])
            col.plot([a[0][0], a[1][0]], [a[0][1], a[1][1]])
            col.plot(xy_temp[idx][0], xy_temp[idx][1], 'bo', markersize=10)
            col.plot(a[1][0], a[1][1], 'ro', markersize=10)
            col.set_xlim([-3, 3])
            col.set_ylim([-3, 3])
    plt.show()