from dataclasses import dataclass
from numbers import Number
from typing import List


@dataclass
class Serie:
    title: str
    data: List[Number]
