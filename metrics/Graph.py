from dataclasses import dataclass
from typing import List

from metrics import Serie


@dataclass
class Graph:
    title: str
    precision: int
    series: List[Serie]
    xaxis: str
    yaxis: str
