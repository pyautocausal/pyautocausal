from typing import Protocol, runtime_checkable, Optional
import networkx as nx
from ..persistence.output_config import OutputConfig



@runtime_checkable
class Node(Protocol):
    """Protocol defining the interface for nodes in the executable graph"""
    name: str
    output: Optional[object]
    output_config: OutputConfig
    
    def is_running(self) -> bool:
        """Returns True if the node is currently running"""
        ...
    
    def is_completed(self) -> bool:
        """Returns True if the node has completed execution"""
        ...
    
    def execute(self) -> None:
        """Execute the node's operation"""
        ... 