import numpy as np


def map_array(arr: np.ndarray, dictionary: dict) -> np.ndarray:
    u, inv = np.unique(arr, return_inverse=True)
    mapped = np.array([dictionary[x] for x in u])[inv].reshape(arr.shape)
    return mapped
