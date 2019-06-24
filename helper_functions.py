import numpy as np
import h5py
import matplotlib.pyplot as plt    
    
def load_dataset(h5py_file,set_type,get_class=False):
    """ 
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
    classes = np.array(file[str('list_classes')][:]) # list of classes,

    if get_class:
        return set_x, set_y, classes
    else:
        return set_x, set_y
