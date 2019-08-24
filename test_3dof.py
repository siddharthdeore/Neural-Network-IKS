import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import plot_model

from utilities import absolute_file_path


dataset_train = np.loadtxt(absolute_file_path('datasets/trains.csv'), delimiter=",")
x_predict=dataset_train[:,:2]
y_predict=dataset_train[:,2:]


# Load precompiled model
model = load_model(absolute_file_path('models/model_s.h5'))
model.summary()
solution=model.predict(x_predict,verbose=2)

ex=solution[:,0]-y_predict[:,0]
ey=solution[:,1]-y_predict[:,1]
err=ex*ex+ey*ey
plt.plot(ex,ey)
#plt.plot(y_predict[:,0],y_predict[:,1],'bo')
plt.show()

plt.plot(err,'ro')
plt.show()

del model  # deletes the existing model