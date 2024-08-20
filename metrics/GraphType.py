from enum import Enum


class GraphType(Enum):
    EMPTY = ""
    LINE = "line"
    COLUMN = "column"

    def __html__(self):
        return self.value
