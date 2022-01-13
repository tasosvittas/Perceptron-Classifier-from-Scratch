import csv
import io
import random
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from numpy import genfromtxt
from numpy.core.numeric import ones
from numpy.lib.function_base import meshgrid

#function energopoihshs = u
def activation(y):
    u = 0
    if y > 0:
        u = 1
        return u
    elif y <= 0:
        u = -1
        return u

with io.open('dataset3D.txt') as csvfile:
    data = csvfile.read()
    csvfile.close()

Data = np.genfromtxt(StringIO(data), dtype=str, delimiter=',')
Data = np.where(Data == 'C1',-1.0, Data)
Data = np.where(Data == 'C2',1.0,Data)

Dataset_float = np.asfarray(Data, dtype = float)
 
#afairesh tou category
category_array = Dataset_float[:,3]
print('Category: ',category_array)

Dataset_float = np.delete(Dataset_float,3,1)

#prosthiki 1 sthn prwth sthlh tou pinaka Dataset_float 
oness = np.ones((Dataset_float.shape[0],1))
ones_array = np.hstack((oness,Dataset_float))
print('Array With Bias: ',ones_array)

w1 = np.random.uniform(-1,1)
w2 = np.random.uniform(-1,1)
w3 = np.random.uniform(-1,1)
w4 = np.random.uniform(-1,1)
weight = [w1,w2,w3,w4]
weight = np.asfarray(weight, dtype=float)
print('Weight: ',weight)

epoch = 50
current_epoch = 0
epoch_counter = 0
learning_rate = 1
current_line= []
current_line =np.asfarray(current_line,dtype=float)

#dhmiourgia 3d figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')   

while current_epoch < epoch:
    print('Current Epoch: ',current_epoch)
    for i in range(len(ones_array)):
        current_line = ones_array[i,:]
        output = ones_array[i,1]*weight[1]+ones_array[i,2]*weight[2]+ones_array[i,3]*weight[3]+ones_array[i,0]*weight[0]
        #output = np.matmul(bias,weight)
        current_line = current_line.reshape(4,1)
        out = activation(output)
        if out != category_array[i]:
            ax.set_xlabel('X axis') 
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis') 
            if out == 1 and category_array[i] == -1:
                weight = weight.reshape(4,1)   
                weight = weight - learning_rate * current_line
                ax.scatter3D(ones_array[:,1],ones_array[:,2],ones_array[:,3],marker='o',c='r')
                xg=[-2,5]
                yg=[-1,1]
                [X,Y]=meshgrid(xg,yg)
                Z=(-weight[0,:]-weight[1,:]*X-weight[2,:]*Y)/weight[3,:]
                ax.plot_surface(X,Y,Z)
                plt.pause(1)
                ax.cla()
            if out == -1 and category_array[i] == 1:
                weight = weight.reshape(4,1)
                weight = weight + learning_rate * current_line
                ax.scatter3D(ones_array[:,1],ones_array[:,2],ones_array[:,3],marker='o',c='r')
                xg=[-2,5]
                yg=[-1,1]
                [X,Y]=meshgrid(xg,yg)
                Z=(-weight[0,:]-weight[1,:]*X-weight[2,:]*Y)/weight[3,:]
                ax.plot_surface(X,Y,Z)
                plt.pause(1)
                ax.cla()
        if out == category_array[i]:
            epoch_counter = epoch_counter + 1
    if epoch_counter == len(category_array):
        print('Finished!')
        break
    current_epoch = current_epoch + 1
    epoch_counter = 0

plt.figure(1)
ax.scatter3D(ones_array[:,1],ones_array[:,2],ones_array[:,3],marker='o',c='r')
xg=[-2,5]
yg=[-1,1]
[X,Y]=meshgrid(xg,yg)
Z=(-weight[0,:]-weight[1,:]*X-weight[2,:]*Y)/weight[3,:]
ax.plot_surface(X,Y,Z)
plt.draw()
plt.show()
