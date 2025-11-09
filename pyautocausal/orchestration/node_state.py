from enum import Enum, auto
from typing import Set

class NodeState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PASSED = "passed"  # New state for nodes that are skipped due to decision branching
    
    @classmethod
    def terminal_states(cls) -> Set['NodeState']:
        return {cls.COMPLETED, cls.FAILED, cls.PASSED}
    
    def is_terminal(self) -> bool:
        return self in self.terminal_states()

