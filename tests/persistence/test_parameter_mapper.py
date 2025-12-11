import pytest
from pyautocausal.persistence.parameter_mapper import make_transformable, TransformableFunction
import inspect
from typing import Dict, List, Any, Optional

def test_basic_transformation():
    """Test basic parameter name transformation"""
    def add(x: int, y: int) -> int:
        return x + y
    
    # Transform parameter names
    transformed = make_transformable(add).transform({'first': 'x', 'second': 'y'})
    
    # Test with transformed names
    result = transformed(first=5, second=3)
    assert result == 8
    
    # Verify signature was updated
    sig = inspect.signature(transformed)
    assert 'first' in sig.parameters
    assert 'second' in sig.parameters

def test_preserve_type_annotations():
    """Test that type annotations are preserved after transformation"""
    def process_list(items: List[int]) -> List[int]:
        return [x * 2 for x in items]
    
    transformed = make_transformable(process_list).transform({'input_items': 'items'})
    
    # Check return annotation
    assert transformed.__annotations__['return'] == List[int]
    
    # Check parameter annotation
    sig = inspect.signature(transformed)
    assert sig.parameters['input_items'].annotation == List[int]

def test_function_transformation():
    """Test basic function transformation"""
    def process(data: Dict[str, int]) -> int:
        return sum(data.values())
    
    transformed = make_transformable(process).transform({'input_dict': 'data'})
    
    # Test the transformed function
    result = transformed(input_dict={'a': 1, 'b': 2})
    assert result == 3

def test_decorator_syntax():
    """Test using the decorator syntax directly"""
    @make_transformable
    def multiply(a: int, b: int) -> int:
        return a * b
    
    assert isinstance(multiply, TransformableFunction)
    
    # Should work as a normal function
    assert multiply(2, 3) == 6
    
    # Should be transformable
    transformed = multiply.transform({'factor1': 'a', 'factor2': 'b'})
    assert transformed(factor1=4, factor2=5) == 20

def test_method_decoration():
    """Test that the decorator works with class methods"""
    class TestClass:
        @make_transformable
        def method(self, x: int, y: int) -> int:
            return x + y + 1  # +1 to verify it's using the instance method
    
    obj = TestClass()
    # Direct call should work
    assert obj.method(5, 3) == 9
    
    # Transformation should work
    transformed = obj.method.transform({'a': 'x', 'b': 'y'})
    assert transformed(a=5, b=3) == 9

def test_default_parameters():
    """Test transformation with default parameters"""
    def func(input_data: List[int], multiplier: float = 1.0) -> List[float]:
        return [x * multiplier for x in input_data]
    
    transformed = make_transformable(func).transform({
        'numbers': 'input_data',
        'factor': 'multiplier'
    })
    
    # Test with default multiplier
    result1 = transformed(numbers=[1, 2, 3])
    assert result1 == [1.0, 2.0, 3.0]
    
    # Test with specified factor
    result2 = transformed(numbers=[1, 2, 3], factor=2.0)
    assert result2 == [2.0, 4.0, 6.0]

def test_partial_transformation():
    """Test transforming only some parameters"""
    def process(data: List[int], factor: float = 1.0) -> List[float]:
        return [x * factor for x in data]
    
    # Only transform the 'data' parameter
    transformed = make_transformable(process).transform({'input_list': 'data'})
    
    # Should work with transformed name for data and original name for factor
    result = transformed(input_list=[1, 2, 3], factor=2.0)
    assert result == [2.0, 4.0, 6.0]

def test_empty_arg_mapping():
    """Test with an empty argument mapping"""
    def simple_func(x: int) -> int:
        return x * 2
    
    # Transform with empty mapping should still work like original
    transformed = make_transformable(simple_func).transform({})
    assert transformed(x=5) == 10

def test_none_annotation():
    """Test with None/Any type annotations"""
    def flexible_func(data: Any, option: Optional[str] = None) -> Any:
        if option:
            return f"{data}-{option}"
        return data
    
    transformed = make_transformable(flexible_func).transform({'input': 'data'})
    
    assert transformed(input="test") == "test"
    assert transformed(input="test", option="suffix") == "test-suffix"
    
    # Check signature
    sig = inspect.signature(transformed)
    assert sig.parameters['input'].annotation == Any

def test_variadic_parameters():
    """Test with *args and **kwargs"""
    def variadic_func(x: int, *args, **kwargs) -> Dict:
        result = {'x': x, 'args': args, 'kwargs': kwargs}
        return result
    
    transformed = make_transformable(variadic_func).transform({'value': 'x'})
    
    result = transformed(value=1, extra=True, option="test")
    assert result['x'] == 1
    assert 'extra' in result['kwargs']
    assert result['kwargs']['option'] == "test"

def test_function_metadata_preservation():
    """Test that function metadata (docstring, name) is preserved"""
    def example_func(x: int) -> int:
        """This is a docstring."""
        return x * 2
    
    transformed = make_transformable(example_func).transform({'value': 'x'})
    
    assert transformed.func.__name__ == example_func.__name__

def test_transformed_signature_is_preserved_through_exposure():
    """
    Tests that the custom signature created by `transform` is not lost
    when passed through the `expose_in_notebook` decorator. This prevents
    a regression where the signature would default to (*args, **kwargs).
    """
    def my_function(a: int, b: str = "hello") -> float:
        """A simple function."""
        return float(a + len(b))

    # `transform` applies `expose_in_notebook` internally
    transformed_func = make_transformable(my_function).transform(
        {'first_arg': 'a', 'second_arg': 'b'}
    )
    
    # Verify the signature of the final, decorated function
    sig = inspect.signature(transformed_func)
    
    # Check that parameters are correctly renamed
    assert 'first_arg' in sig.parameters
    assert 'second_arg' in sig.parameters
    assert 'a' not in sig.parameters
    assert 'b' not in sig.parameters
    
    # Check that annotations and defaults are preserved
    assert sig.parameters['first_arg'].annotation == int
    assert sig.parameters['second_arg'].annotation == str
    assert sig.parameters['second_arg'].default == "hello"
    assert sig.return_annotation == float

def test_descriptor_protocol():
    """Test that the descriptor protocol (__get__) works correctly"""
    class TestClass:
        @make_transformable
        def method(self, x: int) -> int:
            return x + self.offset
        
        def __init__(self):
            self.offset = 10
    
    # Test with two different instances to ensure proper binding
    obj1 = TestClass()
    obj1.offset = 5
    
    obj2 = TestClass()
    obj2.offset = 10
    
    # Transform on the class
    transformed1 = obj1.method.transform({'value': 'x'})
    transformed2 = obj2.method.transform({'value': 'x'})
    
    # Should bind properly when called on instances
    assert transformed1(value=10) == 15
    assert transformed2(value=10) == 20

def test_multiple_transformations():
    """Test that multiple transformations work correctly"""
    def combine(a: int, b: int) -> int:
        return a + b
    
    transformable = make_transformable(combine)
    
    # Apply first transformation
    trans1 = transformable.transform({'x': 'a', 'y': 'b'})
    result1 = trans1(x=1, y=2)
    assert result1 == 3
    
    # Apply second transformation to the same base function
    trans2 = transformable.transform({'first': 'a', 'second': 'b'})
    result2 = trans2(first=3, second=4)
    assert result2 == 7
    
    # Original function still works
    assert transformable(5, 6) == 11

def test_callable_behavior():
    """Test that TransformableFunction remains callable"""
    def square(x: int) -> int:
        return x * x
    
    transformable = make_transformable(square)
    
    # Should work as normal function
    assert transformable(5) == 25
    
    # Should work after transformation
    transformed = transformable.transform({'number': 'x'})
    assert transformed(number=5) == 25 