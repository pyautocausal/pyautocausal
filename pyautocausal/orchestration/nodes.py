import networkx as nx
from typing import Callable, Any, Optional
from .node_state import NodeState
from .base import OutputConfig
from .graph import ExecutableGraph
from .result import Result
from ..persistence.output_types import OutputType
from ..persistence.output_config import OutputConfig
from networkx import DiGraph
from .result import Result
import inspect
from ..persistence.type_inference import infer_output_type
from ..persistence.serialization import prepare_output_for_saving
from pyautocausal.utils.logger import get_class_logger
import warnings
import types


class NodeExecutionError(Exception):
    """Generic wrapper for node execution failures."""
    
    def __init__(self, message: str, node_name: str, original_exception: Exception):
        super().__init__(message)
        self.node_name = node_name
        self.original_exception = original_exception
        self.original_exception_type = type(original_exception).__name__
    
    def __str__(self) -> str:
        return f"Error in node '{self.node_name}' ({self.original_exception_type}): {self.original_exception}"


def is_lambda(func):
    """
    Reliably detect if a function is a lambda function.
    
    This is more reliable than just checking isinstance(func, types.LambdaType)
    because nested functions can sometimes be incorrectly identified as lambdas.
    """
    return isinstance(func, types.LambdaType) and func.__name__ == "<lambda>"

class BaseNode:
    def __init__(self, name: str):
        self.name = name
        self.graph = None
    
    def __str__(self):
        return f"BaseNode(name={self.name})"
    
    # Remove the set_graph method that adds the node to the graph
    # Instead, we'll have a method that only sets the graph reference
    def _set_graph_reference(self, graph: ExecutableGraph):
        """Set the graph reference for this node. Should only be called by the graph."""
        if self.graph is not None and self.graph != graph:
            raise ValueError(f"Node {self.name} is already part of a different graph")
        self.graph = graph
    
class Node(BaseNode):
    def __init__(
            self, 
            name: str, 
            action_function: Callable,
            output_config: Optional[OutputConfig] = None,
            save_node: bool = False,
            node_description: str | None = None,
            display_function: Optional[Callable] = None
        ):
        super().__init__(name)
        self.logger = get_class_logger(f"{self.__class__.__name__}_{name}")
        
        # Validate and setup output configuration
        return_annotation = self._get_return_annotation(action_function)
        self._validate_save_configuration(save_node, return_annotation, output_config)
        self.output_config = self._setup_output_config(save_node, return_annotation, output_config)
        
        # Check if action_function is "static" (does not take self or cls as first argument)
        self.validate_action_function_static(name, action_function)
        
        # Initialize node state and attributes
        self.state = NodeState.PENDING
        self.output = Result()
        self.output_to_save = None
        self.predecessor_outputs = {}
        self.action_function = action_function
        self.execution_count = 0
        self.error = None
        self.node_description = node_description if node_description is not None else inspect.getdoc(getattr(action_function, "func", action_function))
        self.display_function = display_function

    def validate_action_function_static(self, name, action_function):
        sig = inspect.signature(action_function)
        params = list(sig.parameters.values())
        if params and params[0].name in ("self", "cls"):
            raise ValueError(
                f"Node {name}: action_function must be a static function (not a method with 'self' or 'cls' as the first argument)."
            )
    
    def _get_return_annotation(self, action_function: Callable) -> Any:
        """Get the return type annotation from the action function"""
        signature = inspect.signature(action_function)
        return signature.return_annotation
    
    def __rshift__(self, other: 'BaseNode') -> tuple[BaseNode, BaseNode]:
        """Implements the >> operator for node wiring with type validation
        NB: This method is only part of the Node class because the >> syntax is cool.
        Args:
            other: The node to wire to
            
        Returns:
            A tuple containing the source and target nodes
            
        
        """
        if self.graph is None:
            raise ValueError("Node must be added to a graph before wiring")
        self.graph.can_wire_nodes(self, other)
        return (self, other)
    
    def _validate_save_configuration(
            self, 
            save_node: bool, 
            return_annotation: Any, 
            output_config: Optional[OutputConfig]
        ) -> None:
        """Validate the saving configuration and type annotations"""
        # If we plan to save the node output we need to know *how* to save it.
        # That information can come from either a return type annotation (we can
        # infer the output type automatically) OR a user-supplied
        # ``output_config`` that explicitly declares an ``output_type``.
        if save_node and return_annotation == inspect.Parameter.empty:
            if output_config is None or output_config.output_type is None:
                raise ValueError(
                    f"Node {self.name}: When save_node=True, the action function must have a return type annotation "
                    "or you must specify an output_config with an explicit output_type.\n\n"
                    "1. Add a return type hint to your function, e.g.:\n"
                    "   def my_function() -> pd.DataFrame:\n"
                    "       return pd.DataFrame(...)\n\n"
                    "2. Or specify an output_config with an output_type, e.g.:\n"
                    "   output_config=OutputConfig(output_type=OutputType.PARQUET)"
                )
        
        if output_config is not None and not save_node:
            raise ValueError(
                f"Node {self.name}: Cannot specify output_config when save_node is False"
            )

    def _setup_output_config(
            self, 
            save_node: bool, 
            return_annotation: Any, 
            output_config: Optional[OutputConfig]
        ) -> Optional[OutputConfig]:
        """Setup the output configuration based on save settings and type annotation"""
        if output_config is None and save_node:
            output_type = infer_output_type(return_annotation)
            return OutputConfig(
                output_type=output_type,
                output_filename=self.name
            )
        return output_config

    def _resolve_function_arguments(self, func: Callable, available_args: dict) -> dict:
        """Bind named predecessors first, then legacy single inputs and context.

        An explicit binding is never reassigned to a different parameter. The
        historical single-input shorthand remains available for unbound edges.
        """
        signature = inspect.signature(func)
        parameters = {
            name: parameter for name, parameter in signature.parameters.items()
            if parameter.kind not in (inspect.Parameter.VAR_KEYWORD,
                                      inspect.Parameter.VAR_POSITIONAL)
        }
        required = {name for name, parameter in parameters.items()
                    if parameter.default is inspect.Parameter.empty}
        has_varkeyword = any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                             for parameter in signature.parameters.values())
        explicit = self.graph.get_node_explicit_argument_names(self) if self.graph is not None else set()
        unknown_explicit = (set(available_args) & explicit) - set(parameters)
        if unknown_explicit and not has_varkeyword:
            raise ValueError(f"Explicit bindings do not match parameters in node {self.name}: "
                             f"{sorted(unknown_explicit)}")

        arguments = {name: value for name, value in available_args.items()
                     if name in parameters}
        unused = {name: value for name, value in available_args.items()
                  if name not in parameters}
        # Preserve the convenient f(data) <- node_named_something_else form.
        # A named optional parameter must not also fill a required parameter.
        if len(available_args) == 1 and len(unused) == 1:
            source = next(iter(unused))
            target = None
            if source not in explicit:
                if len(required) == 1:
                    target = next(iter(required))
                elif is_lambda(func) and len(parameters) == 1:
                    target = next(iter(parameters))
            if target is not None:
                arguments[target] = unused.pop(source)

        if has_varkeyword:
            arguments.update(unused)
        context = getattr(self.graph, 'run_context', None)
        if context is not None:
            for name in parameters.keys() - arguments.keys():
                if hasattr(context, name):
                    arguments[name] = getattr(context, name)

        missing_required = required - arguments.keys()
        if missing_required:
            function_name = getattr(func, '__name__', type(func).__name__)
            raise ValueError(
                f"Missing required parameters for {function_name} in node {self.name}: "
                f"{missing_required}. Not found in available arguments or run context."
            )
        return arguments

    def _try_recreate_exception_with_node_context(self, original_exception: Exception) -> Optional[Exception]:
        """Attempt to recreate the original exception with node context added."""
        try:
            # Import DataValidationError here to avoid circular imports
            from ..data_validation.validator_base import DataValidationError
            
            # Special handling for DataValidationError
            if isinstance(original_exception, DataValidationError):
                enhanced_message = f"Error in node '{self.name}': {original_exception.args[0]}"
                return DataValidationError(enhanced_message, original_exception.validation_result).with_traceback(original_exception.__traceback__)
            
            # For other exceptions, try to recreate with single message parameter
            exception_type = type(original_exception)
            
            # Get the constructor signature
            sig = inspect.signature(exception_type.__init__)
            params = list(sig.parameters.keys())[1:]  # Skip 'self'
            
            # Check if it's a simple single-argument constructor
            if len(params) == 1:
                param_name = params[0]
                # Common parameter names for message
                if param_name in ('message', 'msg', 'args'):
                    enhanced_message = f"Error in node '{self.name}': {original_exception}"
                    return exception_type(enhanced_message).with_traceback(original_exception.__traceback__)
            
            # If we can't safely recreate, return None to use generic wrapper
            return None
            
        except Exception:
            # If anything goes wrong during recreation, return None to fall back to generic wrapper
            return None

    def execute(self):
        """Execute the node's action function"""
        self.execution_count += 1
        try:
            self.mark_running()
            self._execute()
            self.mark_completed()
        except Exception as e:
            self.mark_failed()
            self.error = {"type": type(e).__name__, "message": str(e)}
            for key in ('cleaning_metadata', 'cleaning_history'):
                if hasattr(e, key):
                    self.error[key] = getattr(e, key)
            
            # Try to preserve original exception type with enhanced message
            enhanced_exception = self._try_recreate_exception_with_node_context(e)
            
            if enhanced_exception is not None:
                raise enhanced_exception
            else:
                # Fall back to generic wrapper that preserves all info
                raise NodeExecutionError(
                    f"Error in node '{self.name}': {e}",
                    self.name,
                    e
                ).with_traceback(e.__traceback__)

    def mark_running(self):
        self.state = NodeState.RUNNING
        
    def mark_completed(self):
        self.state = NodeState.COMPLETED
        
    def mark_failed(self):
        self.state = NodeState.FAILED
        
    def mark_passed(self):
        """Mark this node as passed (skipped due to decision branching)."""
        if self.state == NodeState.PENDING:
            self.state = NodeState.PASSED
            self.logger.info(f"Node {self.name} marked as PASSED (skipped due to decision branching)")
        
    def is_completed(self):
        return self.state == NodeState.COMPLETED
    
    def is_running(self):
        return self.state == NodeState.RUNNING

    def is_passed(self):
        """Returns True if the node has been marked as passed (skipped due to decision branching)."""
        return self.state == NodeState.PASSED

    def get_result_value(self) -> Any:
        """Return the output value of this node from the result dictionary.
        
        This is a convenience method that returns the node's output value
        directly from the result_dict using the node's name as the key.
        
        Returns:
            The output value of this node
        """
        if self.output is None or self.output.result_dict is None:
            return None
        return self.output.result_dict.get(self.name)

    def _execute_action_function(self):
            # Get predecessor outputs directly from the graph
        predecessor_outputs = self.graph.get_node_predecessor_outputs(self)
        
        arguments = self._resolve_function_arguments(self.action_function, predecessor_outputs)
        return (
            self.action_function(**arguments) if arguments 
            else self.action_function()
        )

    def _execute(self):
        """Execute the node's action function with predecessor outputs"""
        output = self._execute_action_function()
        
        self.output.update({self.name: output})
        if self.output_config:
            self.output_to_save = prepare_output_for_saving(output, self.output_config.output_type)
            
    
class DecisionNode(Node):
    """
    A node that makes a decision based on a condition and controls flow to downstream nodes.
    
    DecisionNode evaluates its condition and determines which downstream paths
    should be traversable. Nodes can be designated as "execute when true" (EWT)
    or "execute when false" (EWF).
    """
    def __init__(
            self, 
            name: str,
            condition: Callable,
            node_description: str | None = None
        ):
        # Create a passthrough function that aggregates predecessor outputs
        # by unioning the results of all predecessors
        def passthrough_function(**kwargs):
            result = {}
            for key, value in kwargs.items():
                result.update({key: value})
            return result
        
        super().__init__(
            name=name,
            action_function=passthrough_function,
            output_config=None,
            save_node=False,
            node_description=node_description if node_description is not None else inspect.getdoc(getattr(condition, "func", condition))
        )
        self.condition = condition
        # warn if the condition does not return a boolean
        if not callable(condition):
            raise ValueError(f"Condition for decision node {self.name} is not a callable function")
        # get the signature of the condition
        self.condition_signature = inspect.signature(condition)
        
        # warn if the condition does not have a return type annotation or returns a non-boolean value
        if self.condition_signature.return_annotation is inspect.Parameter.empty:
            warnings.warn(f"Condition for decision node {self.name} does not have a return type annotation")
        elif self.condition_signature.return_annotation != bool:
            warnings.warn(f"Condition for decision node {self.name} returns a non-boolean value: {self.condition_signature.return_annotation}")


        self.decision_result = None
        self._ewt_nodes = set()  # Nodes to execute when condition is True
        self._ewf_nodes = set()  # Nodes to execute when condition is False
    
    def add_execute_when_true(self, node):
        """Add a node to execute when condition evaluates to True"""
        self._ewt_nodes.add(node)
        return self
    
    def add_execute_when_false(self, node):
        """Add a node to execute when condition evaluates to False"""
        self._ewf_nodes.add(node)
        return self
    
    def validate_branches(self):
        """
        Validate that all successors have been classified as either EWT or EWF.
        This should be called before execution.
        """
        if not hasattr(self, 'graph') or self.graph is None:
            return  # Can't validate without a graph
            
        all_successors = set(self.graph.successors(self))

        # validate that successors are not classified as both EWT and EWF
        if self._ewt_nodes.intersection(self._ewf_nodes):
            raise ValueError(
                f"Decision node '{self.name}' has successors that are classified as both execute_when_true and execute_when_false: "
                f"{[node.name for node in self._ewt_nodes.intersection(self._ewf_nodes)]}"
            )

        all_classified = self._ewt_nodes.union(self._ewf_nodes)
        
        missing = all_successors - all_classified
        if missing:
            raise ValueError(
                f"Decision node '{self.name}' has successor nodes that are not classified as "
                f"execute_when_true or execute_when_false: {[node.name for node in missing]}"
            )
        
        extra = all_classified - all_successors
        if extra:
            raise ValueError(
                f"Decision node '{self.name}' has nodes classified as execute_when_true or "
                f"execute_when_false that are not successors: {[node.name for node in extra]}"
            )
    
    def condition_satisfied(self) -> bool:
        """Check if the node's condition is satisfied using predecessors and run context"""
        # Get predecessor outputs directly from the graph
        predecessor_outputs = self.graph.get_node_predecessor_outputs(self)
        
        try:
            # Resolve arguments including run context
            arguments = self._resolve_function_arguments(self.condition, predecessor_outputs)
            condition_result = self.condition(**arguments)
            if not isinstance(condition_result, bool):
                raise ValueError(f"Condition for node {self.name} returned a non-boolean value: {condition_result}")
            return condition_result
        
        except Exception as e:
            raise ValueError(f"Error evaluating condition for node {self.name}: {str(e)}")
    
    def _execute(self):
        """Execute the decision logic and control flow to downstream nodes"""
        

        # Validate that all successors are classified
        self.validate_branches()

        # executing the passthrough function merges all predecessor outputs into a single result
        self.output = Result(super()._execute_action_function())
        
        # Evaluate the condition
        condition_result = self.condition_satisfied()
        self.decision_result = condition_result

        # Log the decision outcome
        self.logger.info(
            f"Decision node '{self.name}': condition evaluated to {condition_result}"
        )
        
        # Set traversability based on condition outcome, 
        for successor in self.graph.successors(self):
            edge = self.graph.edges[self, successor]
            
            if condition_result:
                if successor in self._ewt_nodes:
                    edge['traversable'] = True # this is not strictly necessary, since edges are traversable by default
                else:
                    edge['traversable'] = False
            elif not condition_result:
                if successor in self._ewf_nodes:
                    edge['traversable'] = True
                else:
                    edge['traversable'] = False
            else:
                raise ValueError(f"Decision node '{self.name}' received an unexpected condition result: {condition_result}")
        
        # Mark unreachable nodes as passed
        self.mark_unreachable_nodes_as_passed()
        
    def mark_unreachable_nodes_as_passed(self):
        """Propagate this decision's recorded outcome without re-evaluating it."""
        for successor in self.graph.successors(self):
            if not self.graph.edges[self, successor].get('traversable', True):
                successor.mark_passed()
                self._mark_downstream_as_passed(successor)

    def _mark_downstream_as_passed(self, node):
        for successor in self.graph.successors(node):
            if successor.state != NodeState.PENDING:
                continue
            if all(pred.is_passed() or not self.graph.edges[pred, successor].get('traversable', True)
                   for pred in self.graph.predecessors(successor)):
                successor.mark_passed()
                self._mark_downstream_as_passed(successor)

class InputNode(Node):
    """A node that accepts external input and passes it to its successors."""
    
    def __init__(self, name: str, input_dtype: type = Any):
        self.input_dtype = input_dtype
        super().__init__(name, action_function=lambda: None)
        self.input_set = False
    
    def set_input(self, value: Any):
        """Set the input value that will be passed to successor nodes"""
        if self.input_set:
            raise ValueError(f"Input value for node '{self.name}' has already been set")
        
        if self.input_dtype is not Any and not isinstance(value, self.input_dtype):
            raise TypeError(f"Input value for node '{self.name}' must be of type {self.input_dtype.__name__}, got {type(value).__name__}")
        
        # set the action function in the superclass
        self.action_function = lambda: value
        self.input_set = True

    def _execute(self):
        if not self.input_set:
            raise ValueError(f"Input value for node '{self.name}' has not been set")
        super()._execute()


