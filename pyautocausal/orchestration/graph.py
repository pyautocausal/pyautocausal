from typing import Set, Optional, Dict, Any, Callable, Union, List
from .base import Node
from .result import Result
from .node_state import NodeState

from ..persistence.output_handler import OutputHandler
from pyautocausal.utils.logger import get_class_logger
from pyautocausal.orchestration.run_context import RunContext
from inspect import Parameter, Signature
from ..persistence.local_output_handler import LocalOutputHandler
from pathlib import Path
from ..persistence.output_config import OutputConfig
import networkx as nx
import inspect
import logging
from networkx import DiGraph


class ExecutableGraph(nx.DiGraph):
    def __init__(self):
        """Initialize an empty ExecutableGraph with just the graph structure.
        
        Runtime configuration (output handlers, run context) should be set
        separately using configure_runtime() before execution.
        """
        super().__init__()
        self.logger = get_class_logger(self.__class__.__name__)
        self._input_nodes = {}
        self._nodes_by_name = {}  # New dictionary to track nodes by name
        
        # Runtime configuration - not serialized
        self.run_context = None
        self.output_handler = None
        self.save_node_outputs = False

    def configure_runtime(
        self,
        output_handler: Optional[OutputHandler] = None,
        output_path: Optional[Union[Path, str]] = None,
        run_context: Optional[RunContext] = None
    ) -> 'ExecutableGraph':
        """Configure runtime settings for graph execution.
        
        This method should be called before executing the graph to set up
        output handling and execution context. These settings are not
        persisted when the graph is serialized.
        
        Args:
            output_handler: Custom output handler for saving node outputs
            output_path: Path for default LocalOutputHandler (mutually exclusive with output_handler)
            run_context: Runtime context for execution
            
        Returns:
            self for method chaining
        """
        if output_path is not None and output_handler is not None:
            raise ValueError("Cannot provide both output_path and output_handler")

        self.run_context = run_context or RunContext()
        self.save_node_outputs = False
        
        if output_handler is not None:
            self.output_handler = output_handler
            self.save_node_outputs = True
        elif output_path is not None:
            if isinstance(output_path, str):
                output_path = Path(output_path)
            self.output_handler = LocalOutputHandler(output_path)
            self.save_node_outputs = True
            self.logger.info(f"Local output handler created with path {output_path}")
        else:
            self.logger.warning("No output handler provided, node outputs will not be saved")
            
        return self

    @property
    def input_nodes(self) -> Dict[str, 'InputNode']:
        """Get dictionary of input nodes"""
        return self._input_nodes

    def get_ready_nodes(self) -> Set[Node]:
        """Returns all nodes that are ready to be executed"""
        return {node for node in self.nodes() 
               if isinstance(node, Node) and self.is_node_ready(node)}
    
    def get_running_nodes(self) -> Set[Node]:
        """Returns all nodes that are currently running"""
        return {node for node in self.nodes() 
               if isinstance(node, Node) and node.is_running()}
    
    def is_execution_finished(self) -> bool:
        """Returns True if all nodes in the graph are completed"""
        return self.get_incomplete_nodes() == set()

    def get_incomplete_nodes(self) -> Set[Node]:
        """Returns all nodes that haven't reached a terminal state yet"""
        return {node for node in self.nodes() 
                if isinstance(node, Node) and not node.state.is_terminal()}
    
    def save_node_output(self, node: Node):
        """Save node output if configured to do so"""
        from .nodes import InputNode  # Import here to avoid circular import
        if self.save_node_outputs and self.output_handler is not None and ~isinstance(node, InputNode): # Input nodes are not saved
            if (
                getattr(node, 'output_config', None) is not None
                and node.output is not None
            ):
                output_filename = getattr(node.output_config, 'output_filename', node.name)
                if node.output_config.output_type is None:
                    raise ValueError(f"Output type is not set for node {node.name}")
                self.output_handler.save(output_filename, node.output_to_save, node.output_config.output_type)
            else:
                self.logger.warning(f"Node {node.name} output not saved because no output config was provided")
            
    def execute_graph(self):
        from .nodes import InputNode
        """Execute all nodes in the graph using a breadth-first approach."""
        # Find all start nodes (input nodes or nodes with no predecessors)
        start_nodes = [node for node in self.nodes() 
                      if isinstance(node, Node) and not list(self.predecessors(node))]
        
        if not start_nodes:
            self.logger.warning("No start nodes found in graph")
            return
        
        # Initialize a queue with start nodes
        queue = start_nodes.copy()
        
        # Process nodes in BFS order
        while queue:
            current_node = queue.pop(0)
            
            # Skip if already processed
            if current_node.state.is_terminal():
                continue
            
            # Execute regular node
            current_node.execute()
            self.save_node_output(current_node)
            
            # If completed successfully, mark outgoing edges as traversable
            if current_node.is_completed():
                # Add nodes that are now ready to the queue
                for successor in self.successors(current_node):
                    if self._is_node_ready_to_queue(successor) and successor not in queue:
                        queue.append(successor)
                    
        # Check if all nodes are in terminal state
        incomplete = self.get_incomplete_nodes()
        if incomplete:
            self.logger.warning(
                f"Graph execution completed with {len(incomplete)} incomplete nodes: "
                f"{[node.name for node in incomplete]}"
            )

    def fit(self, **kwargs):
        """
        Set input values and execute the graph.
        
        Args:
            **kwargs: Dictionary mapping input node names to their values
        """
        # Get only input nodes without predecessors
        external_inputs = {
            name: node 
            for name, node in self.input_nodes.items() 
            if not list(self.predecessors(node))
        }
        
        # Validate inputs
        missing_inputs = set(external_inputs.keys()) - set(kwargs.keys())
        if missing_inputs:
            raise ValueError(f"Missing values for input nodes: {missing_inputs}")
        
        extra_inputs = set(kwargs.keys()) - set(external_inputs.keys())
        if extra_inputs:
            raise ValueError(f"Received values for non-existent input nodes: {extra_inputs}")
        
        # Validate and set input values
        for name, value in kwargs.items():
            input_node = external_inputs[name]
            input_node.set_input(value)  # Type checking happens in set_input
        
        # Execute the graph
        self.execute_graph()
        return self

    def get(self, name: str) -> Node:
        """Get a node by its name.
        
        Args:
            name: Name of the node to find
            
        Returns:
            The node with the given name
            
        Raises:
            ValueError: If no node exists with the given name
        """
        node = self._nodes_by_name.get(name)
        if node is None:
            raise ValueError(f"No node found with name '{name}'")
        return node

    def create_node(
        self,
        name: str,
        action_function: Callable,
        predecessors: Optional[Dict[str, str]] = None,
        output_config: Optional[OutputConfig] = None,
        save_node: bool = False,
        node_description: str | None = None,
        display_function: Optional[Callable] = None
    ):
        """
        Add a node to the graph.
        
        Args:
            name: Name of the node
            action_function: Function to execute
            predecessors: Dict mapping argument names to predecessor node names
            condition: Optional callable that returns a boolean 
              and determines if node should execute
            output_config: Optional configuration for node output
            save_node: Whether to save the node's output
            
        Returns:
            self for method chaining
        """
        # Create the node
        from .nodes import Node as NodeObject
        node = NodeObject(
            name=name,
            action_function=action_function,
            output_config=output_config,
            save_node=save_node,
            display_function=display_function
        )
        
        # Use add_node to handle the rest
        return self.add_node_with_predecessors( node, predecessors)
    
    def add_node_to_graph(self, node, **attr):
        """Add a node to the graph and set the graph reference on the node."""
        if not hasattr(node, 'name'):
            raise ValueError(f"Node must have a name, got {type(node)}")
        
        if node.name in self._nodes_by_name:
            raise ValueError(
                f"Cannot add node: a node with name '{node.name}' already exists in the graph"
            )
            
        # Check if node is already in another graph
        if hasattr(node, 'graph') and node.graph is not None and node.graph != self:
            raise ValueError(
                f"Cannot add node '{node.name}': node is already part of a different graph"
            )
        
        # Set the graph reference on the node
        node._set_graph_reference(self)

        # Add the node to the graph
        self._nodes_by_name[node.name] = node
        super().add_node(node, **attr)


    # Override the original add_node to prevent direct addition
    def add_node(self, node, **attr):
        """Override to ensure nodes are added through add_node_to_graph."""
        if hasattr(node, '_set_graph_reference'):
            # This is a BaseNode or subclass, so redirect to add_node_to_graph
            return self.add_node_to_graph(node, **attr)
        else:
            raise ValueError(f"Node must be a BaseNode or subclass, got {type(node)}")

    def create_input_node(self, name: str, input_dtype: type = Any):
        """Add an input node to the graph."""
        from .nodes import InputNode as InputNodeObject
        input_node = InputNodeObject(name=name, input_dtype=input_dtype)
        self.add_node_to_graph(input_node)
        self._input_nodes[name] = input_node
        return self

    def merge_with(self, other: 'ExecutableGraph', *wirings) -> 'ExecutableGraph':
        """Merge another graph into this one with explicit wiring.
        
        Args:
            other: The graph to merge into this one
            *wirings: Variable number of wiring tuples created by the >> operator
            
        Example:
            g1.merge_with(g2, 
                node1 >> input_node1,
                node2 >> input_node2
            )
        """
        from .nodes import InputNode, Node as NodeObject, DecisionNode

        # Validate states first
        non_pending_nodes_self = [
            node.name for node in self.nodes() 
            if hasattr(node, 'state') and node.state != NodeState.PENDING
        ]

        non_pending_nodes_other = [             
            node.name for node in other.nodes() 
            if hasattr(node, 'state') and node.state != NodeState.PENDING
        ]
        if non_pending_nodes_self:
            raise ValueError(
                "Cannot merge graphs: the following nodes in the source graph "
                f"are not in PENDING state: {non_pending_nodes_self}"
            )
        if non_pending_nodes_other:
            raise ValueError(
                "Cannot merge graphs: the following nodes in the target graph "
                f"are not in PENDING state: {non_pending_nodes_other}"
            )

        # Validate wirings
        if not wirings:
            raise ValueError(
                "At least one wiring (e.g., node1 >> input_node2) must be provided "
                "to ensure graphs are connected"
            )

        targets = dict()
        # Validate that all wirings are between the two graphs
        for wiring in wirings:
            source, target = wiring
            if source.graph is not self or target.graph is not other:
                raise ValueError(
                    f"Invalid wiring: {source.name} >> {target.name}. "
                    "Source must be from the left graph and target from the right graph"
                )
            if not isinstance(target, InputNode):
                raise ValueError(
                    f"Invalid wiring: {source.name} >> {target.name}. "
                    "Target must be an InputNode"
                )
            if target in targets:
                raise ValueError(
                    f"Cannot add wiring: {source.name} >> {target.name}. "
                    "Target node already has a wiring"
                )
            targets.update({target: source})
        
        # Create mapping of old nodes to new nodes
        node_mapping = {}
        
        # Re-initialize nodes from other graph
        for node in other.nodes():
            # Generate unique name if there's a conflict
            new_name = node.name
            counter = 1
            while new_name in self._nodes_by_name:
                new_name = f"{node.name}_{counter}"
                counter += 1
            
            if new_name != node.name:
                self.logger.info(
                    f"Renaming node '{node.name}' to '{new_name}' during merge to avoid name conflict"
                )
            
            if isinstance(node, InputNode):
                if node in targets:
                    def make_pass_function(dtype, param_name):
                        def pass_input(**kwargs):
                            return kwargs[param_name]
                        
                        # Create explicit signature with one keyword-only parameter
                        pass_input.__signature__ = Signature([
                            Parameter(param_name, Parameter.KEYWORD_ONLY, annotation=dtype)
                        ])
                        pass_input.__annotations__ = {param_name: dtype, 'return': dtype}
                        return pass_input
                    
                    new_node = NodeObject(
                        name=new_name,
                        # New action function that passes the input from the source node
                        action_function=make_pass_function(node.input_dtype, targets[node].name),
                    )
                    self.add_node_to_graph(new_node)
                else:
                    self.create_input_node(new_name, node.input_dtype)
                node_mapping[node] = self.get(new_name)
            elif isinstance(node, DecisionNode):
                new_node = DecisionNode(
                    name=new_name,
                    condition=node.condition,
                )
                self.add_node_to_graph(new_node)
                node_mapping[node] = new_node
            elif isinstance(node, Node):
                new_node = NodeObject(
                    name=new_name,
                    action_function=node.action_function,
                    output_config=node.output_config,
                    save_node=bool(node.output_config)
                )
                self.add_node_to_graph(new_node)
                node_mapping[node] = new_node
                
            else:
                raise ValueError(f"Invalid node type: {type(node)}")

        # Add edges from the original graph
        for u, v, data in other.edges(data=True):
            if u in node_mapping and v in node_mapping:  # Only add edges between nodes we've mapped
                new_u = node_mapping[u]
                new_v = node_mapping[v]
                self.add_edge(new_u, new_v, **data)

        # Add the wiring edges
        for wiring in wirings:
            source, target = wiring
            new_target = node_mapping[target]
            self.add_edge(source, new_target)

        return self 

    def to_text(self) -> str:
        """Generate a text representation of the graph for debugging.
        
        Returns:
            A string containing a text visualization of the graph structure.
        """
        from .nodes import InputNode
        output = []
        
        # Header
        output.append("Graph Structure:")
        output.append("=" * 50)
        
        # Nodes section
        output.append("\nNodes:")
        output.append("-" * 20)
        for node in sorted(self._nodes_by_name.values(), key=lambda n: n.name):
            node_type = "InputNode" if isinstance(node, InputNode) else "Node"
            state = f"[{node.state.value}]" if hasattr(node, "state") else ""
            output.append(f"{node.name} ({node_type}) {state}")
        
        # Edges section
        output.append("\nConnections:")
        output.append("-" * 20)
        for u, v, data in sorted(self.edges(data=True), key=lambda x: (x[0].name, x[1].name)):
            arg_name = f" as '{data.get('argument_name')}'" if data.get('argument_name') else ""
            output.append(f"{u.name} -> {v.name}{arg_name}")
        
        # Input nodes section
        output.append("\nExternal Inputs:")
        output.append("-" * 20)
        for name, node in sorted(self.input_nodes.items()):
            preds = list(self.predecessors(node))
            if not preds:  # Only show external inputs
                dtype = getattr(node, 'input_dtype', 'Any').__name__
                output.append(f"{name} (expects {dtype})")
        
        return "\n".join(output)

    def print_graph(self):
        """Print a text representation of the graph."""
        print(self.to_text())

    def get_node_predecessors(self, node) -> set:
        """Returns all predecessor nodes for the given node.
        
        Args:
            node: The node whose predecessors to retrieve
            
        Returns:
            Set of predecessor nodes
        """
        return set(self.predecessors(node))

    def get_node_successors(self, node) -> set:
        """Returns all successor nodes for the given node.
        
        Args:
            node: The node whose successors to retrieve
            
        Returns:
            Set of successor nodes
        """
        return set(self.successors(node))

    def get_node_predecessor_outputs(self, node) -> dict:
        """Get outputs from immediate predecessor nodes of the given node.
        
        Args:
            node: The node whose predecessor outputs to retrieve
            
        Returns:
            Dictionary mapping predecessor node names to their outputs
        """
        predecessors = self.get_node_predecessors(node)

        all_outputs = {}
        for predecessor in predecessors:
            if predecessor.is_completed():
                # Node has completed normally, use its output
                if not isinstance(predecessor.output, Result):
                    raise ValueError(f"Predecessor {predecessor.name} output is not a Result")
                all_outputs.update(predecessor.output.result_dict)
            elif predecessor.is_passed():
                # Skip PASSED nodes - they don't contribute outputs
                self.logger.info(f"Skipping PASSED node {predecessor.name} as a predecessor of {node.name}")
                # Don't add any outputs for this node
                continue
            else:
                # Unexpected state
                raise ValueError(
                    f"Predecessor {predecessor.name} is not in COMPLETED or PASSED state, "
                    f"current state: {predecessor.state}"
                )
        return all_outputs

    def is_node_ready(self, node) -> bool:
        """Check if a node is ready to be executed.
        
        A node is ready when:
        1. It is in PENDING state
        2. All incoming edges are traversable and from completed/passed nodes
        
        Args:
            node: The node to check
            
        Returns:
            True if the node is ready to be executed, False otherwise
        """
        if node.state != NodeState.PENDING:
            return False
        
        # Check all incoming edges
        for pred in self.predecessors(node):
            edge = self.edges[pred, node]
            # If any predecessor is not completed/passed or edge is not traversable, node is not ready
            if not (pred.is_completed() or pred.is_passed()) or not edge.get('traversable', True):
                return False
        
        return True

    def validate_node_graph_consistency(self):
        """Validate that all nodes in the graph have this graph as their graph reference."""
        inconsistent_nodes = []
        for node in self.nodes():
            if hasattr(node, 'graph'):
                if node.graph != self:
                    inconsistent_nodes.append(node.name)
        
        if inconsistent_nodes:
            raise ValueError(
                f"Graph inconsistency detected: the following nodes do not reference this graph: "
                f"{inconsistent_nodes}"
            )
        return True

    def can_wire_nodes(self, source: 'BaseNode', target: 'BaseNode') -> bool:
        """Tests if one node can be wired to another node.
        
        Args:
            source: The source node that will output data
            target: The target node that will receive data
            
        Returns:
            bool: True if nodes can be wired, False otherwise
            
        Raises:
            ValueError: If target is not an InputNode
            TypeError: If there is a type mismatch between nodes
        """
        from .nodes import InputNode
        import inspect
        from typing import Any
        import warnings
        
        if not isinstance(target, InputNode):
            raise ValueError(f"Target node must be an input node, got {type(target)}")

        # Get return type from action function signature
        return_annotation = inspect.signature(source.action_function).return_annotation

        # Get expected input type from InputNode
        input_type = target.input_dtype

        # Warn if types cannot be validated
        if return_annotation == inspect.Parameter.empty:
            warnings.warn(
                f"Cannot validate connection: {source.name} -> {target.name}. "
                f"Node {source.name}'s action function lacks return type annotation."
            )
            return True
        elif input_type == Any:
            warnings.warn(
                f"Cannot validate connection: {source.name} -> {target.name}. "
                f"Input node {target.name} accepts Any type."
            )
            return True
        # Validate types if both are specified
        elif return_annotation != inspect.Parameter.empty and input_type != Any:
            if not issubclass(return_annotation, input_type):
                raise TypeError(
                    f"Type mismatch in connection {source.name} -> {target.name}: "
                    f"Node outputs {return_annotation.__name__}, but input node expects {input_type.__name__}"
                )
        return True
    
    def add_node_with_predecessors(self, node, predecessors: List[str]):
        """
        Add a node to the graph with connections to predecessor nodes.
        
        Args:
            node: Node to add
            predecessors: Dict mapping argument names to predecessor node names
            
        Returns:
            The added node
        """
        if predecessors is None:
            predecessors = {}
        # First check if all predecessors are in the graph
        for pred_name in predecessors:
            if pred_name not in self._nodes_by_name:
                raise ValueError(f"Predecessor node {pred_name} not found in graph")
        
        # Add node to graph
        self.add_node_to_graph(node)

        # Add predecessors if specified
        if predecessors:
            for pred_name in predecessors:
                # throws error if predecessor not found
                pred_node = self.get(pred_name)
                self.add_edge(pred_node, node)
        
        return self
    
    def add_edge(self, u, v, **attr):
        """Add an edge from u to v with traversable=False by default."""
        attr['traversable'] = True
        super().add_edge(u, v, **attr)

    def _is_node_ready_to_queue(self, node) -> bool:
        """Check if a node is ready to be added to the execution queue.
        
        A node is ready when:
        1. It is in PENDING state
        2. All incoming edges are from completed/passed nodes and are traversable
        
        Args:
            node: The node to check
            
        Returns:
            True if the node is ready to be queued, False otherwise
        """
        if node.state != NodeState.PENDING:
            return False
        
        # Check all incoming edges
        for pred in self.predecessors(node):
            edge = self.edges[pred, node]
            # If any predecessor is not completed/passed or edge is not traversable, node is not ready
            if not (pred.is_completed() or pred.is_passed()) or not edge.get('traversable', True):
                return False
        
        return True

    def create_decision_node(
        self,
        name: str,
        condition: Callable,
        predecessors: Optional[List[str]] = None,
        save_node: bool = False,
    ):
        """
        Add a decision node to the graph.
        
        Args:
            name: Name of the node
            condition: function that returns a boolean
            predecessors: List of predecessor node names
            output_config: Optional configuration for node output
            save_node: Whether to save the node's output
            
        Returns:
            self for method chaining
        """
        from .nodes import DecisionNode
        
        node = DecisionNode(
            name=name,
            condition=condition,
        )
        
        # Use add_node to handle the rest
        return self.add_node_with_predecessors(node, predecessors)

    def when_true(self, decision_node_name: str, ewt_node_name: str):
        """
        Define nodes that should execute when a decision node's condition is true.
        
        Args:
            decision_node_name: Name of the decision node
            ewt_node_name: Name of the node to execute when the condition is true
        """
        decision_node = self.get(decision_node_name)
        from .nodes import DecisionNode
        
        if not isinstance(decision_node, DecisionNode):
            raise ValueError(f"Node '{decision_node_name}' is not a DecisionNode")
        
        decision_node.add_execute_when_true(self.get(ewt_node_name))
        return self

    def when_false(self, decision_node_name, ewf_node_name):
        """
        Define nodes that should execute when a decision node's condition is false.
        
        Args:
            decision_node_name: Name of the decision node
            ewf_node_name: Name of the node to execute when the condition is false
        """
        decision_node = self.get(decision_node_name)
        from .nodes import DecisionNode
        if not isinstance(decision_node, DecisionNode):
            raise ValueError(f"Node '{decision_node_name}' is not a DecisionNode")
        decision_node.add_execute_when_false(self.get(ewf_node_name))
        return self

    def __getstate__(self):
        """Custom serialization to exclude runtime configuration."""
        state = self.__dict__.copy()
        # Remove runtime configuration that shouldn't be serialized
        state['run_context'] = None
        state['output_handler'] = None
        state['save_node_outputs'] = False
        # Remove logger as it's not serializable
        if 'logger' in state:
            del state['logger']
        return state

    def __setstate__(self, state):
        """Custom deserialization to restore object state."""
        self.__dict__.update(state)
        # Restore logger
        self.logger = get_class_logger(self.__class__.__name__)
        # Runtime configuration will be None and needs to be set via configure_runtime()

    # save/load removed; use to_yaml / from_yaml instead.

    def to_yaml(self, filepath: Union[str, Path]) -> Path:
        """Serialise this graph to a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Destination where the YAML should be written.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        from pyautocausal.persistence.yaml_codec import graph_to_yaml  # Local import to avoid heavy deps at module import time
        return graph_to_yaml(self, filepath)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "ExecutableGraph":
        """Load a graph from a YAML file previously produced by :py:meth:`to_yaml`."""
        from pyautocausal.persistence.yaml_codec import yaml_to_graph
        loaded = yaml_to_graph(filepath)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is type {type(loaded)}, expected ExecutableGraph")
        return loaded