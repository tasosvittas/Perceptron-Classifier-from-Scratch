import csv
import io
from os import PRIO_USER
import random
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from numpy import genfromtxt
from numpy.core.numeric import ones
from numpy.core.shape_base import block

#function energopoihshs = u
def activation(y):
    u = 0
    if y > 0:
        u = 1
        return u
    elif y <= 0:
        u = -1
        return u

with io.open('dataset2D.txt') as csvfile:
    data = csvfile.read()
    csvfile.close()

Data = np.genfromtxt(StringIO(data), dtype=str, delimiter=',')
Data = np.where(Data == 'C1',-1.0, Data)
Data = np.where(Data == 'C2',1.0,Data)

Dataset_float = np.asfarray(Data, dtype = float)

#afairesh tou category
category_array = Dataset_float[:,2]
print('Category: ',category_array)

Dataset_float = np.delete(Dataset_float,2,1)

#prosthiki 1 sthn prwth sthlh tou pinaka Dataset_float 
oness = np.ones((Dataset_float.shape[0],1))
ones_array = np.hstack((oness,Dataset_float))
print('Array With Bias: ',ones_array)

w1 = np.random.uniform(-1,1)
w2 = np.random.uniform(-1,1)
w3 = np.random.uniform(-1,1)
weight = [w1,w2,w3]
weight = np.asfarray(weight, dtype=float)
print('Weight: ',weight)

epoch = 50
current_epoch = 0
epoch_counter = 0
learning_rate = 1
current_line = []
current_line =np.asfarray(current_line,dtype=float)
plt.scatter(ones_array[:,1], ones_array[:,2],marker='o',c='r')
while current_epoch < epoch:
    print('Current Epoch: ',current_epoch)
    for i in range(len(ones_array)):
        current_line = ones_array[i,:]
        output = ones_array[i,1]*weight[1]+ones_array[i,2]*weight[2]+weight[0]*ones_array[i,0]
        current_line = current_line.reshape(3,1)
        #output = np.matmul(bias,weight)
        out = activation(output)
        if out != category_array[i]:
            if out == 1 and category_array[i] == -1:
                weight = weight.reshape(3,1)   
                weight = weight - learning_rate * current_line
                x1 = weight[1,:]/weight[2,:]
                x2 = weight[0,:]/weight[2,:]
                y=-x1*ones_array-x2
                plt.scatter(ones_array[:,1], ones_array[:,2],marker='o',c='r')
                plt.plot(ones_array,y,'k')
                plt.axis([0,1.5,0,2])
                plt.draw()
                plt.pause(1)
                plt.clf()
            if out == -1 and category_array[i] == 1:
                weight = weight.reshape(3,1)
                weight = weight + learning_rate * current_line
                x1 = weight[1,:]/weight[2,:]
                x2 = weight[0,:]/weight[2,:]
                y=-x1*ones_array-x2
                plt.scatter(ones_array[:,1], ones_array[:,2],marker='o',c='r')
                plt.plot(ones_array,y,'k')
                plt.axis([0,1.5,0,2])
                plt.draw()
                plt.pause(1)
                plt.clf()
        if out == category_array[i]:
            epoch_counter = epoch_counter + 1
    if epoch_counter == len(category_array):
        print('Finished!')
        break
    current_epoch = current_epoch + 1
    epoch_counter = 0

weight = weight.reshape(3,1)
plt.scatter(ones_array[:,1], ones_array[:,2],marker='o',c='r')
x1 = weight[1,:]/weight[2,:]
x2 = weight[0,:]/weight[2,:]
y=-x1*ones_array-x2
plt.plot(ones_array,y,'k')
plt.show()

    
    
    

    
    
    
    



    