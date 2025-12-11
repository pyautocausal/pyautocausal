"""
PyAutoCausal Example Graph (Legacy Interface)

This module maintains backward compatibility by importing from the new 
organized structure in the example_graph/ directory.

For new code, prefer importing directly from:
    from pyautocausal.pipelines.example_graph import causal_pipeline, simple_graph, main
"""

# Import everything from the new organized structure
from .example_graph import causal_pipeline, simple_graph, main

# Maintain backward compatibility
__all__ = [
    'causal_pipeline',
    'simple_graph', 
    'main'
]

# Make main executable when run directly
if __name__ == "__main__":
    main()
