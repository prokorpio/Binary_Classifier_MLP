import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from helper_functions import *


## Load Dataset
train_set_x_orig, train_set_y, classes = \
    load_dataset('datasets/train_catvnoncat.h5','train_set',get_class=True)
test_set_x_orig, test_set_y = \
    load_dataset('datasets/test_catvnoncat.h5','test_set')

## Dataset Pre-processing
# flatten
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T 
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    # reminder: wrong use of reshape got you stuck for two days...
# normalize wrt max value
train_set_x = train_set_x/255   
test_set_x = test_set_x/255     
#train_seet_x shaped (12288,209)
#test_set_x shaped (12288,50)
#train_set_y shaped (1,209)
#test_set_y shaped (1,50)

# save dataset
#np.savetxt('train_set_x.txt',train_set_x, fmt='%1.9f')
#np.savetxt('train_set_y.txt',train_set_y, fmt='%1.9f')
#np.savetxt('test_set_x.txt',test_set_x, fmt='%1.9f')
#np.savetxt('test_set_y.txt',test_set_y, fmt='%1.9f')

## Initialize Model
layer_dims = [train_set_x.shape[0], 20, 7, 5, 1]
parameters = initialize_parameters(layer_dims)

## Train Model
num_of_layers = len(layer_dims) # including input layer
hidden_act_type = 'relu'
output_act_type = 'sigmoid'
num_of_iterations = 1500
costs = [] # list of cost values

for i in range(num_of_iterations): 
    input_data = train_set_x
    caches = [] # list with each element being a tuple (A_prev,W,Z) of each layer
    grads = {}  # dictionary of dW and db for each layer
    
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
    caches.append(cache) # caches will have indeces [0, .. ,num_of_layers-2]
                         # because for loop started with 1...

                
    # Compute Cost (for entire train set examples)
    cost = compute_cost(train_set_y, y_hat, \
                            loss_type='binary_cross_entropy') 
       

    # Backward Prop
    # for output layer:
    AL = y_hat
    Y = train_set_y.reshape(AL.shape)
    dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # dA of output layer,
                                         #dependent on activation and loss type
    A_prev, W, Z = caches[num_of_layers-2] 
    dA_prev, dW, db = backward_layer(dA, Z, A_prev, W, output_act_type) 
    grads['dW' + str(num_of_layers - 1)] = dW
    grads['db' + str(num_of_layers - 1)] = db
    # for hidden layers:
    for l in reversed(range(1,num_of_layers-1)):    # starts at l = num-2  
        dA = dA_prev
        A_prev, W, Z = caches[l-1] 
        dA_prev, dW, db = backward_layer(dA, Z, A_prev, W, hidden_act_type)
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db


    # Gradient Descent
    for l in range(1,num_of_layers):
        optimize(parameters['W'+str(l)], grads['dW'+str(l)])
        optimize(parameters['b'+str(l)], grads['db'+str(l)])

    
    # Print Cost
    if i % (num_of_iterations//10) == 0:
        costs.append(cost)
        print("Cost after iteration {}: {}".format(i,np.squeeze(cost)))

## Save Parameters
#np.save('parameters.npy',parameters) 
                
## Print Train Set Accuracy
predictions = np.int64(y_hat > 0.5)
print("Train Set Accuracy = ", \
    str(np.sum((predictions== train_set_y)/y_hat.shape[1])))


## Print Cost Plot
#plt.plot(np.squeeze(costs))
#plt.ylabel('cost')
#plt.xlabel('iterations (per hundreds)')
#plt.show()  

                
## Predict
_ = predict(parameters, test_set_x, test_set_y, 'relu','sigmoid')



## End of code
