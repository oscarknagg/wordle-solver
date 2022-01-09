import numpy as np


def map_array(arr: np.ndarray, dictionary: dict) -> np.ndarray:
    u, inv = np.unique(arr, return_inverse=True)
    mapped = np.array([dictionary[x] for x in u])[inv].reshape(arr.shape)
    return mapped


def nunique_per_row(arr: np.ndarray):
    assert len(arr.shape) == 2
    sorted_arr = np.sort(arr, axis=1)
    nunique = (sorted_arr[:, 1:] != sorted_arr[:, :-1]).sum(axis=1) + 1
    return nunique
