import numpy as np

def save_np(arr, path):
    np.save(path, arr)

def load_np(path):
    return np.load(path)
