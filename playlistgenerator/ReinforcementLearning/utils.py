import numpy as np


def softmax(array: np.array) -> np.array:
    """Performs a softmax over current song choices to represent user selection.

    Args:
        array (np.array): Current possible song selections.

    Returns:
        np.array: Probabilities of users liking the selections.
    """
    exp_x = np.exp(array - np.max(array))
    return exp_x / exp_x.sum(axis=0)


def small_hash(num: str) -> int:
    """Creates a short hash from a string, to avoid breaking DQN model limits.

    Args:
        num (str): Number to hash

    Returns:
        int: Shortened hashed version of number
    """
    return abs(hash(num)) % 1000
