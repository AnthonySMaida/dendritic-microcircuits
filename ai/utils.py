from typing import Iterable, TypeVar

import numpy as np

T = TypeVar('T')


def iter_with_prev(it: Iterable[T]):
    last_element = None
    for e in it:
        yield last_element, e
        last_element = e


def create_column_vector(*values):
    return np.array(list(map(lambda x: [x], values)))
