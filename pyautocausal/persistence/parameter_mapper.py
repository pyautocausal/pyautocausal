from typing import Callable, Dict, Any, TypeVar, ParamSpec, Optional
from functools import wraps
import inspect
from pyautocausal.persistence.notebook_decorators import expose_in_notebook
from inspect import Signature

P = ParamSpec('P')
R = TypeVar('R')

class TransformedFunctionWrapper:
    """
    A simple wrapper that holds a function and its argument mapping.
    This is what gets pickled, ensuring the __signature__ is not lost.
    """
    def __init__(self, func: Callable, arg_mapping: Dict[str, str]):
        self.func = func
        self.arg_mapping = arg_mapping

    def __call__(self, *args, **kwargs):
        # This is the actual execution logic
        mapped_kwargs = {}
        for external_name, func_param in self.arg_mapping.items():
            if external_name in kwargs:
                mapped_kwargs[func_param] = kwargs.pop(external_name)
        
        return self.func(*args, **{**kwargs, **mapped_kwargs})

class TransformableFunction:
    """Base class for functions that support method chaining for parameter renaming."""
    
    def __init__(self, func: Callable[P, R]):
        self._func = func
        # Store the original function's signature and return annotation
        self._signature = inspect.signature(func)
        self._return_annotation = self._signature.return_annotation
    
    def get_function(self) -> Callable[P, R]:
        return self._func

    def transform(self, arg_mapping: Optional[Dict[str, str]] = None) -> Callable[P, R]:
        """
        Transform the function's parameter names using the provided mapping.
        
        Args:
            arg_mapping: Dictionary mapping external parameter names to function parameter names
                        e.g., {'node_output': 'func_param'} means node output named 'node_output' 
                        maps to this function's parameter 'func_param'
        
        Returns:
            A wrapped function with renamed parameters
        """
        if not arg_mapping:
            return self.get_function()
        # Create a reverse mapping for parameter inspection
        reverse_mapping = {v: k for k, v in arg_mapping.items()}
        
        # Create the simple wrapper that will be pickled
        wrapper = TransformedFunctionWrapper(self._func, arg_mapping)
        
        # Preserve the return type annotation on the simple wrapper
        wrapper.__annotations__ = {'return': self._return_annotation}
        
        # Store the argument mapping on the wrapper function for debugging
        wrapper._arg_mapping = arg_mapping
        
        # Create a customized inspect.signature for the wrapper function
        original_sig = self._signature
        parameters = list(original_sig.parameters.values())
        
        # Replace parameter names according to the reverse mapping
        # This is critical for _resolve_function_arguments to work correctly
        modified_parameters = []
        for param in parameters:
            if param.name in reverse_mapping:
                # Create a new parameter with the external name
                from inspect import Parameter
                modified_param = Parameter(
                    name=reverse_mapping[param.name],
                    kind=param.kind,
                    default=param.default,
                    annotation=param.annotation
                )
                modified_parameters.append(modified_param)
            else:
                modified_parameters.append(param)
        
        # Create a new signature with the modified parameters
        
        modified_sig = Signature(
            parameters=modified_parameters,
            return_annotation=original_sig.return_annotation
        )
        
        # Attach the modified signature to the wrapper function
        wrapper.__signature__ = modified_sig
        
        # Apply expose_in_notebook to the wrapper
        return expose_in_notebook(self._func, arg_mapping=arg_mapping)(wrapper)
    
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Make the TransformableFunction callable, returning original function result"""
        return self._func(*args, **kwargs)
    
    def __get__(self, obj: Any, objtype: Any = None) -> 'TransformableFunction':
        """Support for method decorators."""
        if obj is None:
            return self
        return TransformableFunction(self._func.__get__(obj, objtype))

def make_transformable(func: Callable[P, R]) -> TransformableFunction:
    """Decorator to make a function or method support parameter transformation."""
    return TransformableFunction(func) 