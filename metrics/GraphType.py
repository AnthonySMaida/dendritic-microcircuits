from enum import Enum


class GraphType(Enum):
    LINE = "line"
    COLUMN = "column"

    def __html__(self):
        return self.value
