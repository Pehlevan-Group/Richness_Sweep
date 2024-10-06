from collections import defaultdict
import numpy as np

def default_factory(): 
   return defaultdict(list)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def generate_logspace(length, dtype=int):
    """ Generate an almost logspaced sequence of the form
    [0, 1, 2, 5, 10, 20, 50, ... ] etc
    """
    base_values = np.array([1, 2, 5])
    exponent_values = np.logspace(0, length // 3, num=length // 3+1, base=10, dtype=int)
    result = np.outer(exponent_values, base_values).flatten()
    result = np.insert(result, 0, 0)
    return result[:length].astype(dtype)
