from dataclasses import dataclass
from typing import List, Optional

from metrics import Serie
from metrics.GraphType import GraphType


@dataclass
class Graph:
    type: GraphType
    title: str
    precision: int
    series: List[Serie]
    xaxis: Optional[str] = None
    """Optional x-axis label (not used with column type)"""
    yaxis: Optional[str] = None
    """Optional y-axis label"""
    categories: Optional[List[str]] = None
    """Optional list of categories for the x-axis (column type only)"""
    caption: Optional[str] = None
    """Optional caption for the graph"""
    extra: Optional[dict] = None

    @staticmethod
    def empty() -> "Graph":
        return Graph(GraphType.EMPTY, "", 1, [])
