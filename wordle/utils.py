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


def softmax(logits: np.ndarray, temperature: float = 1.0):
    exp_x = np.exp((logits - logits.max()) / temperature)
    probs = exp_x / exp_x.sum()
    return probs


if __name__ == '__main__':
    x = np.array([1.0, 1.0, 2.0])
    print(softmax(x))
    print(softmax(x, 0.1))
    print(softmax(x, 10))
