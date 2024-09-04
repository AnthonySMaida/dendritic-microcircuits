from typing import Iterable, TypeVar

import numpy as np

from ai.config import Config

T = TypeVar('T')


def iter_with_prev(it: Iterable[T]):
    last_element = None
    for e in it:
        yield last_element, e
        last_element = e


def create_column_vector(*values):
    return np.array(list(map(lambda x: [x], values)))


def logsig(x, alpha=None):
    """
    logistic sigmoid
    soft ReLU is another possibility (supplementary data p. 3)

    :param x: input value
    :param alpha: affects slope near orig. Recommend alpha >= 1 to avoid shallow grad.
    :return:
    """
    alpha = alpha or Config.alpha
    return 1.0 / (1.0 + np.exp(-alpha * x))
