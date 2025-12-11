from typing import Any

class Result:
    """A class that represents the result of a node execution.
       A result class is a thin wrapper around a dictionary where
       the key is the name of the node and the value is the output of the node.

       For typical nodes, the result class will have one key, which is the name of the node.
       Decision nodes pass on all of their predecessors' results as outputs.
       As such, the result class for a decision node will have one key for each predecessor and 
       one value for each predecessors outputs.
    """
    def __init__(self, result_dict: dict | None = None):
        self.result_dict = result_dict or {}

    def __getitem__(self, key: str) -> Any:
        return self.result_dict[key]

    def __setitem__(self, key: str, value: Any):
        self.result_dict[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.result_dict
    
    def union(self, other: 'Result') -> 'Result':
        self.result_dict.update(other.result_dict)
        return self
    
    def is_empty(self) -> bool:
        return not self.result_dict
    
    def update(self, other: dict) -> None:
        self.result_dict.update(other)

    def get_only_item(self) -> Any:
        """Get the only item in the result dictionary.
        Raises a ValueError if the result dictionary is empty or has more than one item."""
        if len(self.result_dict) != 1:
            raise ValueError("Result dictionary must have exactly one item")
        return list(self.result_dict.values())[0]

