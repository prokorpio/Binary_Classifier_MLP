import numpy as np
import matplotlib.pyplot as plt
import h5py
from helper_function.py import *

## Load Dataset
train_set_x_orig, train_set_y_orig, classes = \
    load_dataset('datasets/train_catvnoncat.h5','train_set',get_class=True)
test_set_x_orig, test_set_y_orig= \
    load_dataset('datasets/test_catvnoncat.h5','test_set')

## Dataset Pre-processing
# flatten
train_set_x = train_set_x_orig.reshape((-1,train_set_x_orig.shape[0]))
train_set_y = train_set_y_orig.reshape((-1,train_set_x_orig.shape[0]))
test_set_x = test_set_x_orig.reshape((-1,test_set_x_orig.shape[0]))
test_set_y = test_set_y_orig.reshape((-1,test_set_x_orig.shape[0]))
# normalize wrt max value
train_set_x = train_set_x/255
train_set_y = train_set_y/255 
test_set_x = test_set_x/255
test_set_y = test_set_y/255
# print values
#print(train_set_x.shape)
#print(train_set_y.shape)
#print(test_set_x.shape)
#print(test_set_y.shape)

## Initialize Model
layer_dims = [train_set_x.shape[0], 5, 3, 1]
parameters = initialize_parameters(layer_dims)

## Train Model
num_of_layers = len(layer_dims)
caches = [] # list with each element being a tuple (A_prev,Z) of each layer
hidden_act_type = 'relu'
output_act_type = 'sigmoid'
input_data = train_set_x

# Forward Prop
# for hidden layers:     
for l in range(1,num_of_layers-1):  
    cache, A = forward_layer(input_data, parameters['W'+str(l)], \
                                  parameters['b'+str(l)], hidden_act_type)
    caches.append(cache)   # append tuple to list
    input_data = A 
# for output layer:     
cache, y_hat = forward_layer(input_data, parameters['W'+str(num_of_layers-1)],\
                         parameters['b'+str(num_of_layers-1)], output_act_type)
caches.append(cache)

# Compute Cost (for entire train set)
cost = compute_cost(train_set_y, y_hat, \
                        activation_type='binary_cross_entropy') 
   

## Backward Prop
# for output layer:
grads = {}  # dictionary of dW and db for each layer
Y = train_set_y
AL = y_hat
dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # dA of output layer,
                                     #dependent on activation and loss type
A_prev, W, Z = caches[-1] 
dA_prev, dW, db = backward_layer(dA, Z, A_prev, W, output_act_type) 
grads["dW" + str(num_of_layers - 1)] = dW
grads["db" + str(num_of_layers - 1)] = db

# for hidden layers:
for l in reversed(range(1,num_of_layers-1)):    # starts at l = num_.. - 2  
    dA = dA_prev
    A_prev, W, Z = caches[l] 
    dA_prev, dW, db = backward_layer(dA, Z, A_prev, W, hidden_act_type)
    grads["dW" + str(l)] = dW
    grads["db" + str(l)] = db


    # compute dAL
    # for each of the layers, starting from the output
        #compute dZ
        #compute dW
        #compute db
        #compute dA_prev (dA of the layer to the left)

## Predict
                
