import matplotlib.pyplot as plt
import numpy as np
import random
from math import sin, cos, tan
from math import pi

#random.seed(17)
# Rotation matrix
def rot(ang):
    return np.array([
        [cos(ang),-sin(ang)],
        [sin(ang),cos(ang)]
        ])

# Forward kinematics
def fk(q0,q1,q2):
    R0=rot(q0)
    J1=np.matmul(R0,[[1],[0]])
    R1=rot(q0+q1)
    J2=J1+np.matmul(R1,[[1],[0]])
    R2=rot(q0+q1+q2)
    ee=J2+np.matmul(R2,[[1],[0]])
    return ee,J1,J2

# def plot_fk(q0,q1,q2):
#     ee,J1,J2=fk(q0,q1,q2)
#     x1=J1[0][0]
#     x2=J2[0][0]
#     x3=ee[0][0]
#     y1=J1[1][0]
#     y2=J2[1][0]
#     y3=ee[1][0]
#     plt.plot([0,x1,x2,x3],[0,y1,y2,y3],'b-',x3,y3,'r.')
#     plt.axis([-3, 3, -3, 3])
#     plt.show()

# #plot_fk(pi/2,pi/2,-pi/2)

# x= np.empty((0,3))
# y= np.empty((0,2))

# for i in range (0,360,10):
#     for j in range(0,360,20):
#         for k in range(-60,60,5):
#             q0=i*pi/180
#             q1=j*pi/180
#             q2=(k+random.randint(-30,30))*pi/180
#             ee,J1,J2=fk(q0,q1,q2)
#             x = np.append(x, np.array([[q0,q1,q2]]), axis=0)
#             y = np.append(y, np.array([[ee[0][0],ee[1][0]]]), axis=0)
#             x1=J1[0][0]
#             x2=J2[0][0]
#             x3=ee[0][0]
#             y1=J1[1][0]
#             y2=J2[1][0]
#             y3=ee[1][0]
#             plt.plot([0,x1,x2,x3],[0,y1,y2,y3])
#             #plt.plot(x3,y3,'r.')
# plt.show()
# plt.axis([-3, 3, -3, 3])
# print(x.shape)
# print(y.shape)