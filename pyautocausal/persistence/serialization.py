from typing import Any
from ..persistence.output_types import OutputType

def jsonify(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [jsonify(item) for item in obj]
    elif hasattr(obj, 'to_dict'):  # Handle pandas and similar objects
        return jsonify(obj.to_dict())
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return jsonify(obj.__dict__)
    return obj

def prepare_output_for_saving(output: Any, output_type: OutputType) -> Any:
    """Prepare output for saving based on its type"""
    if output_type == OutputType.JSON:
        return jsonify(output)
    return output 