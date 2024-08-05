from collections.abc import Iterator
from typing import Iterable, TypeVar, Tuple

T = TypeVar('T')


def iter_with_prev(it: Iterable[T]) -> Iterator[Tuple[T | None, T]]:
    last_element = None
    for e in it:
        yield last_element, e
        last_element = e
