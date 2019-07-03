import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from helper_functions import *


## Load Dataset
train_set_x_orig, train_set_y_orig, classes = \
    load_dataset('datasets/train_catvnoncat.h5','train_set',get_class=True)
test_set_x_orig, test_set_y_orig = \
    load_dataset('datasets/test_catvnoncat.h5','test_set')

## Dataset Pre-processing
# flatten
train_set_x = train_set_x_orig.reshape((-1,train_set_x_orig.shape[0])) 
train_set_y = train_set_y_orig.reshape((-1,train_set_x_orig.shape[0]))
test_set_x = test_set_x_orig.reshape((-1,test_set_x_orig.shape[0]))
test_set_y = test_set_y_orig.reshape((-1,test_set_x_orig.shape[0]))
# normalize wrt max value
train_set_x = train_set_x/255   #(12288,209)
#train_set_y shaped (1,209)
test_set_x = test_set_x/255     #(12288,50)
#test_set_y shaped (1,50)
# shorten test set
test_set_x = test_set_x[:,:3]
test_set_y = test_set_y[:,:3]

## Initialize Model
layer_dims = [test_set_x.shape[0], 7, 5, 1]
parameters = initialize_parameters(layer_dims)


## Define functions for Gradient Checkin
epsilon = 1e-7

def dictionary_to_vector(dictionary,w_label,b_label,layer_dims):
    
    keys = []
    for l in range(1,len(layer_dims)):
        weights = dictionary[w_label+str(l)].reshape(-1,1)
        biases = dictionary[b_label+str(l)]   
        if l == 1:
            theta = weights
        else:
            theta = np.concatenate((theta,weights),axis=0)
        theta = np.concatenate((theta,biases),axis=0)
        keys = keys + [w_label+str(l)]*weights.shape[0]
        keys = keys + [b_label+str(l)]*biases.shape[0]

    return np.array(theta), keys

def vector_to_dictionary(vector,layer_dims):
    a = layer_dims[0]
    b = layer_dims[1]
    c = layer_dims[2]
    d = layer_dims[3]
    parameters = {}
    parameters['W1'] = vector[:(b*a)].reshape((b,a))
    parameters['b1'] = vector[b*a:b*a+(b)].reshape((b,1))
    parameters['W2'] = vector[(b*a+b):(b*a+b)+(c*b)].reshape((c,b))
    parameters['b2'] = vector[(b*a+b)+(c*b):(b*a+b)+(c*b)+(c)].reshape((c,1))
    parameters['W3'] = vector[(b*a+b)+(c*b)+(c):(b*a+b)+(c*b)+(c)+(c*d)].reshape((d,c))
    parameters['b3'] = vector[(b*a+b)+(c*b)+(c)+(c*d):(b*a+b)+(c*b)+(c)+(c*d)+(d)].reshape((d,1))
    
    return parameters

## Train Model
num_of_layers = len(layer_dims) # including input layer
hidden_act_type = 'relu'
output_act_type = 'sigmoid'
num_of_iterations = 1
costs = [] # list of cost values

for i in range(num_of_iterations): 
    input_data = test_set_x
    caches = [] # list with each element being a tuple (A_prev,W,Z) of each layer
    grads = {}  # dictionary of dW and db for each layer
    
    # Forward Prop
    # for hidden layers:     
    for l in range(1,num_of_layers-1):  
        print('Forward Prop Executed')
        cache, A = forward_layer(input_data, parameters['W'+str(l)], \
                                      parameters['b'+str(l)], hidden_act_type)
        caches.append(cache)   # append tuple to list
#        print(input_data[0] - A[0])
        input_data = A 
    # for output layer:     
    cache, y_hat = forward_layer(input_data, parameters['W'+str(num_of_layers-1)],\
                             parameters['b'+str(num_of_layers-1)], output_act_type)
    caches.append(cache) # caches will have indeces [0, .. ,num_of_layers-2]
                         # because for loop started with 1...

                
    # Compute Cost (for entire train set examples)
    cost = compute_cost(test_set_y, y_hat, \
                            loss_type='binary_cross_entropy') 
       

    # Backward Prop
    # for output layer:
    AL = y_hat
    Y = test_set_y.reshape(AL.shape)
    dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # dA of output layer,
                                         #dependent on activation and loss type
    A_prev, W, Z = caches[num_of_layers-2] 
    dA_prev, dW, db = backward_layer(dA, Z, A_prev, W, output_act_type) 
    grads['dW' + str(num_of_layers - 1)] = dW
    grads['db' + str(num_of_layers - 1)] = db
    # for hidden layers:
    for l in reversed(range(1,num_of_layers-1)):    # starts at l = num-2  
        print('Backward Prop Executed')
        dA = dA_prev
        A_prev, W, Z = caches[l-1] 
        dA_prev, dW, db = backward_layer(dA, Z, A_prev, W, hidden_act_type)
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db

                
## Actual Gradient Checking
vector_parameters, keys = dictionary_to_vector(parameters,'W','b',layer_dims)               
vector_grads, _ = dictionary_to_vector(grads,'dW','db',layer_dims)
np.savetxt('keys.txt', keys, fmt='%s')   
np.savetxt('grad.txt', vector_grads, fmt='%1.9f')
num_parameters = vector_parameters.shape[0] 
J_plus = np.zeros((num_parameters, 1))
J_minus = np.zeros((num_parameters, 1))
gradapprox = np.zeros((num_parameters, 1))
print("Performing Gradient Check")

for i in range(0,num_parameters):
    
    ## Forward Prop for thetaplus
    # for hidden layers:     
    thetaplus = np.copy(vector_parameters)
    thetaplus[i] += epsilon
    theta_dict = vector_to_dictionary(thetaplus,layer_dims)
    input_data = test_set_x
    for l in range(1,num_of_layers-1):  
#        print('Forward Prop Executed')
        _, A = forward_layer(input_data, theta_dict['W'+str(l)], \
                                      theta_dict['b'+str(l)], hidden_act_type)
#        caches.append(cache)   # append tuple to list
#        print(input_data[0] - A[0])
        input_data = A 
    # for output layer:     
    _, y_hat = forward_layer(input_data, theta_dict['W'+str(num_of_layers-1)],\
                             theta_dict['b'+str(num_of_layers-1)], output_act_type)
#    caches.append(cache) # caches will have indeces [0, .. ,num_of_layers-2]
                         # because for loop started with 1...
                
    # Compute Cost for thetaplus
    J_plus[i] = compute_cost(test_set_y, y_hat, \
                            loss_type='binary_cross_entropy') 

                

    ## Forward Prop for thetaminus
    # for hidden layers:     
    thetaminus = np.copy(vector_parameters)
    thetaminus[i] -= epsilon
    theta_dict = vector_to_dictionary(thetaminus,layer_dims)
    input_data = test_set_x
    for l in range(1,num_of_layers-1):  
#        print('Forward Prop Executed')
        _, A = forward_layer(input_data, theta_dict['W'+str(l)], \
                                      theta_dict['b'+str(l)], hidden_act_type)
#        caches.append(cache)   # append tuple to list
#        print(input_data[0] - A[0])
        input_data = A 
    # for output layer:     
    _, y_hat = forward_layer(input_data, theta_dict['W'+str(num_of_layers-1)],\
                             theta_dict['b'+str(num_of_layers-1)], output_act_type)
#    caches.append(cache) # caches will have indeces [0, .. ,num_of_layers-2]
                         # because for loop started with 1...
                
    # Compute Cost for thetaminus
    J_minus[i] = compute_cost(test_set_y, y_hat, \
                            loss_type='binary_cross_entropy') 
    
    # Compute grad approximation
    gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
    # Show Progress Bar
    if i % (num_parameters//100) == 0:
        print("\r Done: {}%".format(100*(i/num_parameters)),end="",flush=True)
##find significant differences
    #print(vector_grads.shape)
#    if abs(vector_grads[i] - gradapprox[i,0]) > 0.9e-6:
#        print("Possible rror on " + str(keys[i]))
#        print("\t diff = " + str(abs(vector_grads[i] - gradapprox[i])))
##end of additional code

## Compare gradapprox and vector_grads
numerator = np.linalg.norm(vector_grads- gradapprox)                        # Step 1'
denominator = np.linalg.norm(vector_grads) +np.linalg.norm(gradapprox)      # Step 2'
difference = numerator/denominator                                          # Step 3'
np.savetxt('aprox.txt', gradapprox, fmt='%1.9f')

if difference > 2e-7:
    print ("\033[93m" + "\nThere is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
else:
    print ("\033[92m" + "\nYour backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    

## End of code

