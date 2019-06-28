import numpy as np
import matplotlib.pyplot as plt
import h5py


## Load Dataset
def load_dataset(h5py_file, set_type, get_class=False):
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
            np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
            # note that we multiply 0.01 to make the weights very small
            # and we also use randn to use a normal distribution
        parameters['b'+str(i)] = np.zeros((layer_dims[i],1))

    return parameters 

parameters = initialize_parameters([4,5,3,1])

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

def compute_linear(X, W, b, activation):
    """
    Description: 
        A single neuron in the computational graph, computes Z and A
        
    Arguments:
        X = input matrix to the neuron, shaped (n_x,m)
        W = weights of shape (neurons_in_l, neurons_in_prev_l)
        b = biases of shape (neurons_in_l, 1)
        activation = which activation to use: relu,tanh,sigmoid,softmax. 
    
    Outputs:
        A = output matrix of the neuron 
    """
    
    Z = np.dot(W,X) + b

    if activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'tanh':
        A = np.tanh(Z)

    return A

                
def compute_cost(Y, Y_hat, type='binary_cross_entropy'):
    """
    Description:
        Computes different types of costs

    Arguments:
        Y = true labels
        Y_hat = approximation of Y
        type =  type of cost to compute
    Outputs:
        J = scalar cost value
    """
    m = len(Y)
    
    if type == 'binary_cross_entropy':
        J = (-1/m)*(np.dot(Y,np.log(Y_hat).T) + np.dot(1-Y,np.log(1-Y_hat).T))


    return J

                
def forward_prop(X, Y, parameters, hid_activation, out_activation, iterations):
    """
    Description:
        Performs one forward prop through the paramaters

    Arguments:
        X = input data
        Y = true labels
        parameters = dictionary of weights and biases of the network
                   = {W#:matrix,  
                      b#:vector}
        hid_activation = activation in hidden layers
        out_activation = activation in output layer
        iterations = number of iterations
 
    Outputs:
       cost = the cost of the propagation
       cache = cache of computed values (determine which is needed)
    """

# Backward Propagation
    # compute dAL
    # for each of the layers, starting from the output
        #compute dZ
        #compute dW
        #compute db
        #compute dA_prev (dA of the layer to the left)

# Optimize: Gradient Descent
    # update dW
    # update db

# After training, predict
        
