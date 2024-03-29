import numpy as np
import h5py
import matplotlib.pyplot as plt    
   

def load_dataset(h5py_file,set_type,get_class=False):
    """ 
    Description:
        Will load the individual parts of h5py convcat dataset

    Argument: 
        h5py_file = str, dir/name of h5 cat dataset file to read
        set_type = str, 'train_set' or 'test_set'
        get_class = If True, will return classes
    Returns:
        set_x = input feature matrix of shape (m, num_px, num_px,3) 
        set_y = input label vector of shape (1, m)
        classes = meaning of set_y labels
    """

    file = h5py.File(h5py_file,'r')
    set_x = np.array(file[str(set_type) + '_x'][:]) # features   
    set_y = np.array(file[str(set_type) + '_y'][:]) # 1 or 0 labels
    set_y = set_y.reshape(1,set_y.shape[0])
    classes = np.array(file[str('list_classes')][:]) # list of classes

    if get_class:
        return set_x, set_y, classes
    else:
        return set_x, set_y

                
## Create and Initialize Network Parameters
def initialize_parameters(layer_dims):
    """
    Description:
        Will create layer parameters and initialize its values

    Argument:
        layer_dims = a list [n_x, n_h, nL] containing layer dimensions
    Returns:
        parameters = a dictionary of weights and biases 
                   = {W#:matrix,  
                      b#:vector}
    """
    np.random.seed(1)               # to get consistent initial vals 
    num_of_layers = len(layer_dims) # input layer included            
    parameters = {}                 # initialize dictionary 
                
    for i in range(1,num_of_layers):
        parameters['W'+str(i)] = \
            np.random.randn(layer_dims[i],layer_dims[i-1])/ \
        np.sqrt(layer_dims[i-1]) #*0.01
            # note that we multiply 0.01 to make the weights very small
            # and we also use randn to use a normal distribution
        parameters['b'+str(i)] = np.zeros((layer_dims[i],1))


    return parameters 


## Forward Propagation (use cache)
def sigmoid(x):
    """
    Description:
        Performs element-wise sigmoid of x 
                
    Argument:
        x = scalar or numpy array of any size
    Returns:
        y = element-wise sigmoid of the input
    """
                
    y = 1/(1+np.exp(-x)) #implicit broadcasting


    return y

def forward_layer(A_prev, W, b, activation_type):
    """
    Description: 
        A single layer in the forward computational graph, computes Z and A
        
    Arguments:
        A_prev = input matrix to the neuron, shaped (n_x-or-a,m)
        W = weights of shape (neurons_in_l, neurons_in_prev_l)
        b = biases of shape (neurons_in_l, 1)
        activation_type = which activation to use: relu,tanh,sigmoid,softmax. 
    
    Outputs:
        cache = (A_prev, W, Z) where Z is output of the linear function,
                        and A_prev is act_matrix from the left
        A = out of the activation layer
    """
    
    Z = np.dot(W,A_prev) + b

    if activation_type == 'sigmoid':
        A = sigmoid(Z)
    elif activation_type == 'relu':
        A = np.maximum(0,Z)

    cache = (A_prev,W,Z) # tuple            


    return cache, A 

                
                
def compute_cost(Y, Y_hat, loss_type='binary_cross_entropy'):
    """
    Description:
        Computes different types of costs

    Arguments:
        Y = true labels
        Y_hat = approximation of Y
        loss_type =  type of cost to compute
    Outputs:
        J = scalar cost value
    """
    m = Y.shape[1] 
    
    if loss_type == 'binary_cross_entropy':
        J = (-1./m)*(np.dot(Y,np.log(Y_hat).T) + np.dot(1-Y,np.log(1-Y_hat).T))


    return J


# Backward Propagation
def backward_layer(dA,Z,A_prev,W,activation_type):
    """
    Description: 
        Computes a bakward propagation on a single layer:
            dJ/d(W or b) = (dJ/dA)(dA/dZ)(dZ/d(W or b))

    Arguments: 
        dA = partial of the current layer's activation wrt J
        Z = current Z value of the layer, needed to compute dZ
        A_prev = the input matrix for the current layer, needed for dW
        W = layer's weights, needed for dA_prev computation
        activation_type = type of activation to differentiate
    Outputs:
        dW = partial of the layer's W wrt J
        db = partial of the layer's b wrt J
        dA_prev = dA of the layer to the left  
    """

    m = A_prev.shape[1] # num of examples is num of columns
    if activation_type == 'relu':            
        dg_dz = np.int64(Z>0)   # converts T/F matrix of condition to 1/0
                                # derivative of relu wrt z is 1 for z>0
    elif activation_type == 'sigmoid':
        dg_dz = sigmoid(Z)*(1 - sigmoid(Z))

    dZ = dA * dg_dz
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True) 
                        # keepdims will keep the dimention of the axis we
                        # perform the summation in, instead of eliminating it
    dA_prev = np.dot(W.T,dZ) # dA of the layer to the left 
    
    
    return dA_prev, dW, db


# Optimize: Gradient Descent
def optimize(parameter, grad, learning_rate=0.0075):
    """
        Description: 
            Performs parameter (weights or biases) update for a layer

        Arguments: 
            parameter = matrix/vector of weights/biases
            grad = dW or db
            learning_rate = to be used for gradient descent formula
        Outputs:
            none, mutable argument parameter is modified    
             
    """
    # update dW or db
    parameter -= learning_rate*grad     # modify parameter object

        
def predict(parameters, X, Y, hidden_act_type, output_act_type):
    """
        Description:
            Performs prediction on the test set
        Arguments:
            parameters = dict of trained weights and biases
            X = matrix of pre-processed input data
            Y = correct labels of X  
            hidden_act_type = activation function used for hidden layer
            output_act_type = activation function used for output layer
        Outputs:
            predictions = predicted labels for test set

    """

    ## Forward Prop
    num_of_layers = len(parameters)//2
    m = X.shape[1]
    # for hidden layers:     
    input_data = X
    for l in range(1,num_of_layers):  
        _, A = forward_layer(input_data, parameters['W'+str(l)], \
                                      parameters['b'+str(l)], hidden_act_type)
        input_data = A 
    # for output layer:     
    _, y_hat = forward_layer(input_data, parameters['W'+str(num_of_layers)],\
                             parameters['b'+str(num_of_layers)], output_act_type)
    
    predictions = np.int64(y_hat > 0.5)
    print("Test Set Accuracy = " + str(np.sum((predictions == Y)/m)))


    return predictions
