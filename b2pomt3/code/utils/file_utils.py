import numpy as np


def read_data(filename, shape):
    """
    Parameters
    ----------
    filepath : string
       dataset location
    shape : tuple (x,y)
       x is the number of entry in the dataset
       y is the dimension of the entry
    """
    # read binary data and return as a numpy array
    fp = np.memmap(filename, dtype='float32', mode='r', shape=shape)
    data = np.zeros(shape=shape)
    data[:] = fp[:]
    del fp

    return data
