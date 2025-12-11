from typing import Dict, List, Any, Optional, Tuple, Callable
import networkx as nx
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import inspect
import ast
from pathlib import Path
from ..orchestration.nodes import Node, InputNode, DecisionNode
from ..orchestration.graph import ExecutableGraph
from .visualizer import visualize_graph
from .notebook_runner import run_notebook_and_create_html, convert_notebook_to_html
import importlib
from ..persistence.parameter_mapper import TransformableFunction


class NotebookExporter:
    """
    Exports an executed graph to a Jupyter notebook format.
    
    This class handles the conversion of a DAG execution into a linear
    sequence of notebook cells that can be executed sequentially.
    """
    
    def __init__(self, graph: ExecutableGraph):
        self.graph = graph
        self.nb = new_notebook()
        self._var_names: Dict[str, str] = {}  # Maps node names to variable names
        # get module this is run from
        self.this_module = inspect.getmodule(inspect.getouterframes(inspect.currentframe())[1][0])
        self.needed_imports = set()
    def _get_topological_order(self) -> List[Node]:
        """Get a valid sequential order of nodes for the notebook."""
        return list(nx.topological_sort(self.graph))
    
    def _create_header(self) -> None:
        """Create header cell with metadata about the graph execution."""
        header = "# Causal Analysis Pipeline\n\n"
        header += "This notebook was automatically generated from a PyAutoCausal pipeline execution.\n\n"
        
        self.nb.cells.append(new_markdown_cell(header))
    
    def _create_imports_cell(self, cell_index: int) -> None:
        """Create cell with all necessary imports."""
        all_imports = "\n".join(self.needed_imports)
        # insert cell at cell_index
        self.nb.cells.insert(cell_index, new_code_cell(all_imports))
    
    def _is_exposed_wrapper(self, func) -> bool:
        """Check if a function is decorated with expose_in_notebook."""
        return hasattr(func, '_notebook_export_info') and func._notebook_export_info.get('is_wrapper', False)
    
    def _get_exposed_target_info(self, func) -> Tuple[Any, Dict[str, str]]:
        """Get the target function and argument mapping from an exposed wrapper."""
        if not self._is_exposed_wrapper(func):
            return None, {}
        
        info = func._notebook_export_info
        
        # Resolve the target function from its portable path
        path = info.get('target_function_path')
        if not path:
            return None, {}
            
        module_name, qualname = path.split(":", 1)
        try:
            module = importlib.import_module(module_name)
            target_func = getattr(module, qualname)
        except (ImportError, AttributeError) as e:
            # If we can't resolve it, we can't get the source.
            # This might happen if the execution environment is different.
            # We'll log this but proceed without the target source.
            print(f"Warning: Could not resolve target function '{path}': {e}") # Replace with logger
            return None, {}

        return target_func, info.get('arg_mapping', {})
    
    def _get_function_imports(self, func) -> None:
        """Extract import statements needed for a given function."""
        try:
            # First analyze the module to get all its imports and namespace
            module = inspect.getmodule(func)
            
            # Track imports from the module
            import_mapping = {}  # Maps name -> import statement
            import_aliases = {}  # Maps alias -> (module, original_name)
            
            try:
                module_source = inspect.getsource(module)
                module_tree = ast.parse(module_source)
                
                for node in ast.walk(module_tree):
                    if isinstance(node, ast.ImportFrom):
                        module_name = node.module
                        for alias in node.names:
                            name = alias.name
                            asname = alias.asname or name
                            
                            # Store the import statement
                            if alias.asname:
                                import_mapping[asname] = f"from {module_name} import {name} as {asname}"
                            else:
                                import_mapping[name] = f"from {module_name} import {name}"
                            
                            # Store alias mapping
                            import_aliases[asname] = (module_name, name)
                    
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.name
                            asname = alias.asname or name
                            
                            if alias.asname:
                                import_mapping[asname] = f"import {name} as {asname}"
                            else:
                                import_mapping[name] = f"import {name}"
                            
                            import_aliases[asname] = (name, None)
                            
            except Exception:
                # If module parsing fails, fallback to namespace analysis
                pass
            
            # Get function source and parse it
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            # Collect all names used in the function
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    # For attribute access like PanelOLS.from_formula, we need the base name
                    used_names.add(node.value.id)
            
            # For each used name, determine if we need an import
            for name in used_names:
                # Skip built-ins and function parameters
                if name in ['self', 'cls'] or name in dir(__builtins__):
                    continue
                
                # Check if we have this import from the module's import statements
                if name in import_mapping:
                    self.needed_imports.add(import_mapping[name])
                    continue
                
                # Check if it's in the module's namespace
                if hasattr(module, name):
                    obj = getattr(module, name)
                    
                    # If it's a module, import it
                    if inspect.ismodule(obj):
                        self.needed_imports.add(f"import {obj.__name__}")
                    
                    # If it has a module attribute, try to import from there
                    elif hasattr(obj, '__module__') and obj.__module__:
                        obj_module = obj.__module__
                        
                        # Don't import built-ins
                        if obj_module == 'builtins':
                            continue
                            
                        # Try to import from the object's module
                        try:
                            # Check if this is a class/function we can import
                            if (inspect.isclass(obj) or inspect.isfunction(obj) or 
                                callable(obj)):
                                self.needed_imports.add(f"from {obj_module} import {name}")
                        except Exception:
                            # If import fails, try just importing the module
                            self.needed_imports.add(f"import {obj_module}")
        
        except Exception as e:
            # Fallback: just add a comment about the error
            self.needed_imports.add(f"# Import analysis failed: {str(e)}")
    
    def _format_function_definition(self, node: Node) -> str:
        """Format the function definition for a node."""
        if isinstance(node, InputNode):
            return ""

        if isinstance(node, DecisionNode):
            func = node.condition
        else:
            func = node.action_function
        
        # Check if this is an exposed wrapper function
        if self._is_exposed_wrapper(func):
            target_func_or_wrapper, arg_mapping = self._get_exposed_target_info(func)
            
            # If the target is a TransformableFunction, get the actual underlying function
            if isinstance(target_func_or_wrapper, TransformableFunction):
                target_func = target_func_or_wrapper.get_function()
            else:
                target_func = target_func_or_wrapper

            if target_func is None:
                # Handle case where the target function could not be resolved
                return f"# Source code for target function could not be resolved."

            target_source = inspect.getsource(target_func)
            
            # Remove the @make_transformable decorator lines but preserve indentation
            target_lines = target_source.split('\n')
            filtered_lines = []
            for line in target_lines:
                if not line.strip().startswith('@make_transformable'):
                    filtered_lines.append(line)
            target_source = '\n'.join(filtered_lines)

            # Format a comment explaining the wrapper relationship
            mapping_str = ", ".join([f"'{wrapper}' → '{target}'" for wrapper, target in arg_mapping.items()])
            comment = f"# This node uses a wrapper function that calls a target function with adapted arguments\n"
            comment += f"# Argument mapping: {mapping_str}\n\n"
            
            # Get the target function's name
            target_name = target_func.__name__
            
            # Create imports using AST analysis
            self._get_function_imports(target_func)
            
            # Add both the target and wrapper with proper imports
            return comment + target_source
        
        # For regular functions, collect imports
        self._get_function_imports(func)
        
        # Handle lambdas
        source = inspect.getsource(func)
        if source.strip().startswith('lambda'):
            # Get the lambda body
            lambda_body = source.split(':')[1].strip()
            # Create a proper function definition
            source = f"def {node.name}_func(*args, **kwargs):\n    return {lambda_body}"
        else:
            # For regular functions, we need to preserve indentation but handle decorators
            func_lines = source.split('\n')
            processed_lines = []
            
            for i, line in enumerate(func_lines):
                if line.strip().startswith('@'):
                    # Keep decorator lines as-is (without extra indentation)
                    processed_lines.append(line.strip())
                    # Add a newline after the decorator if the next line is the function definition
                    if (i + 1 < len(func_lines) and 
                        func_lines[i + 1].strip().startswith('def ')):
                        # Don't add extra empty line, just ensure the next line comes after a newline
                        pass
                else:
                    # For function definition and other lines, strip leading whitespace 
                    # but preserve the overall structure
                    if line.strip().startswith('def '):
                        processed_lines.append(line.strip())  # Remove all leading whitespace from def line
                    else:
                        # Keep other lines with their original relative indentation
                        processed_lines.append(line)
            
            source = '\n'.join(processed_lines)
        
        return source
    
    def _get_function_name_from_string(self, function_string: str) -> str:
        """Get the function name from a string."""
        return function_string.split('def')[1].split('(')[0].strip()
    
    
    def _find_argument_source_nodes(self, current_node: Node) -> Dict[str, str]:
        """
        Traces backwards from a node to find the non-decision-node ancestors
        that provide data. Effectively finds the origins of data flowing into
        the current_node, ignoring intermediate DecisionNodes.

        Args:
            current_node: The node to start tracing backwards from.

        Returns:
            A dictionary mapping the names of ancestor source node names to the nodes 
            (e.g., {'df': Node, 'settings': Node}).
            This indicates which data sources are potentially available.
        """
        source_nodes = {}
        visited = set()
        queue = list(self.graph.predecessors(current_node)) # Use list for queue behavior

        while queue:
            predecessor = queue.pop(0) # FIFO

            if predecessor in visited:
                continue
            visited.add(predecessor)

            if isinstance(predecessor, DecisionNode):
                # If it's a decision node, add its predecessors to the queue
                # to continue tracing backwards *through* it.
                for decision_predecessor in self.graph.predecessors(predecessor):
                    if decision_predecessor not in visited:
                        queue.append(decision_predecessor)
            else:
                # If it's a regular node or an input node, it's a source.
                # We store its name as an available data source.
                source_nodes[predecessor.name] = predecessor

        return source_nodes

    def _resolve_function_arguments(self, node: Node, func: Callable) -> Dict[str, str]:
        """Resolve the arguments for a node's function."""

        is_wrapper = self._is_exposed_wrapper(func)
        
        arguments = dict()  
        #TODO: Handle default arguments for non-wrapper functions
        

        # For wrapper functions, use the argument mapping to map the arguments to the predecessor node names
        if is_wrapper:
            target_func, arg_mapping = self._get_exposed_target_info(func)
            for predecessor_name, func_param in arg_mapping.items():
                arguments[func_param] = f"{predecessor_name}_output"
        else:
            arg_mapping = dict()

        # Get predecessor nodes that provide data, ignoring decision nodes in between
        arg_source_nodes = self._find_argument_source_nodes(node)        

        # Handle the arguments that are not transformed
        for arg_name, _ in arg_source_nodes.items():
            if arg_name not in arg_mapping:
                arguments[arg_name] = f"{arg_name}_output"

        #TODO: Add check that all required arguments are present and all provided arguments are part of arguments
        #TODO: Handle run-context arguments

        return arguments
    
    def _format_function_execution(self, node: Node, function_string: str) -> str:
        """Format the function execution statement."""
        if isinstance(node, InputNode):
            return f"{node.name}_output = input_data['{node.name}']"

        # TODO: Handle cases where we want to show condition functions
        func = node.action_function

        arguments = self._resolve_function_arguments(node, func)
        
        repr_string_noop = lambda x: repr(x) if not isinstance(x, str) else x
        
        # Format argument string
        args_str = ", ".join(f"{k}={repr_string_noop(v)}" for k, v in arguments.items()) if arguments else ""
        
        
        is_wrapper = self._is_exposed_wrapper(func)
        
        # For wrapped functions, add a comment showing how to call the target directly
        if is_wrapper:
            target_func, arg_mapping = self._get_exposed_target_info(func)
            
            # For target function call, map the arguments according to the mapping
            target_args = {}
            for wrapper_arg, wrapper_value in arguments.items():
                target_arg = arg_mapping.get(wrapper_arg, wrapper_arg)
                target_args[target_arg] = wrapper_value
            
            target_args_str = ", ".join(f"{k}={repr(v)}" for k, v in target_args.items()) if target_args else ""
            
            # Get the target function's name
            if target_func:
                # If the target is a TransformableFunction, get the actual function's name
                if isinstance(target_func, TransformableFunction):
                    target_name = target_func.get_function().__name__
                else:
                    target_name = target_func.__name__
            else:
                target_name = func._notebook_export_info.get('target_function_path', 'unknown_target')
            
            # Create both function calls, with the direct call commented out
            function_name = self._get_function_name_from_string(function_string)
            wrapper_call = f"{node.name}_output = {function_name}({args_str})"
            
            return wrapper_call
        
        # Normal case - just call the function
        function_name = self._get_function_name_from_string(function_string)
        return f"{node.name}_output = {function_name}({args_str})"
    
    def _create_node_cells(self, node: Node) -> None:
        """Create cells for a single node's execution."""
        # Add markdown cell with node info
        info = f"## Node: {node.name}\n"
        if node.node_description:
            info += f"{node.node_description}\n"
        self.nb.cells.append(new_markdown_cell(info))
    
        if not isinstance(node, InputNode):
            func_def = self._format_function_definition(node)
            self.nb.cells.append(new_code_cell(func_def))
        
            exec_code = self._format_function_execution(node, func_def)
            self.nb.cells.append(new_code_cell(exec_code))
        
            if node.display_function:
                notebook_display_code = self._format_display_function_call(node)
                self.nb.cells.append(new_code_cell(notebook_display_code))
    
    def _format_display_function_call(self, node: Node) -> str:
        """Format the display function call for a node. This simply
        calls the display function on the node's output."""
        # Collect imports for the display function
        if node.display_function:
            self._get_function_imports(node.display_function)
        
        return f"# Display Result\n{node.display_function.__name__}({node.name}_output)"

    def _create_input_node_cells(self, node: InputNode, has_data_loading: bool = False) -> None:
        """Create cells for a single input node's execution."""
        # Add markdown cell with node info
        info = f"## Node: {node.name}\n"
        if node.node_description:
            info += f"{node.node_description}\n"
        self.nb.cells.append(new_markdown_cell(info))
        
        if has_data_loading:
            # For data loading, use the loaded data directly
            # Since input_data contains the loaded dataset, assign it directly to the node output
            exec_code = f"{node.name}_output = input_data"
        else:
            # Add execution cell which is just a comment telling the user to provide the input
            exec_code = f"# TODO: Load your input data for '{node.name}' here\n{node.name}_output = None  # Replace with your data"
        
        self.nb.cells.append(new_code_cell(exec_code))

    def _extract_imports_from_loading_function(self, loading_function: str) -> None:
        """Extract and add necessary imports based on the loading function string."""
        # Common patterns for import detection
        import_patterns = {
            'pd.': 'import pandas as pd',
            'pandas.': 'import pandas as pd',
            'np.': 'import numpy as np',
            'numpy.': 'import numpy as np',
            'pickle.': 'import pickle',
            'json.': 'import json',
            'yaml.': 'import yaml',
            'torch.': 'import torch',
            'joblib.': 'import joblib',
        }
        
        for pattern, import_stmt in import_patterns.items():
            if pattern in loading_function:
                self.needed_imports.add(import_stmt)

    def export_notebook(self, filepath: str, data_path: Optional[str] = None, loading_function: Optional[str] = None) -> None:
        """
        Export the graph execution as a Jupyter notebook.
        
        Args:
            filepath: Path where the notebook should be saved
            data_path: Optional path to data file to load
            loading_function: Optional function string to load the data (e.g., 'pd.read_csv')
        """
        # Reset notebook and imports to ensure clean state
        self.nb = new_notebook()
        self.needed_imports = set()
        
        # Create header
        self._create_header()
        
        # Get graph visualization without title
        markdown_content = visualize_graph(self.graph)
        # Remove the title line and empty line after it
        markdown_content = "\n".join(markdown_content.split("\n")[1:])
        self.nb.cells.append(new_markdown_cell(markdown_content))

        # Extract imports from data loading function (if provided)
        if data_path and loading_function:
            self._extract_imports_from_loading_function(loading_function)

        # Process nodes in topological order (this will collect imports as we go)
        for node in self._get_topological_order():
            # Skip decision nodes
            if isinstance(node, DecisionNode):
                continue
                
            # Only export completed nodes
            if node.is_completed():
                if isinstance(node, InputNode):
                    self._create_input_node_cells(node, data_path is not None and loading_function is not None)
                else:
                    self._create_node_cells(node)

        # Now insert imports cell at the right position (after header/visualization, before everything else)
        imports_position = 2  # After header (0) and visualization (1)
        self._create_imports_cell(imports_position)

        # Add data loading cell after imports (if both parameters are provided)
        if data_path and loading_function:
            data_loading_position = imports_position + 1  # Right after imports
            
            # Get input node names to create proper dictionary structure
            input_node_names = [node.name for node in self._get_topological_order() 
                              if isinstance(node, InputNode) and node.is_completed()]
            
            # Always load data directly into input_data (to match test expectations)
            data_loading_code = f"""# Load input data
input_data = {loading_function}('{data_path}')"""
            
            self.nb.cells.insert(data_loading_position, new_code_cell(data_loading_code))
        
        # Save the notebook
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(self.nb, f)
    
    def export_and_run_to_html(
        self,
        notebook_filepath: str | Path,
        html_filepath: Optional[str | Path] = None,
        data_path: Optional[str] = None,
        loading_function: Optional[str] = None,
        timeout: int = 600,
        kernel_name: str = "python3"
    ) -> Path:
        """
        Export the graph as a notebook, execute it, and convert to HTML.
        
        This is a convenience method that combines export_notebook() with 
        run_notebook_and_create_html() to create an executed HTML report.
        
        Args:
            notebook_filepath: Path where the notebook should be saved
            html_filepath: Path for the output HTML file. If None, uses same name as notebook with .html extension
            data_path: Optional path to data file to load
            loading_function: Optional function string to load the data (e.g., 'pd.read_csv')
            timeout: Maximum time in seconds to wait for each cell execution (default: 600)
            kernel_name: Name of the Jupyter kernel to use for execution (default: "python3")
            
        Returns:
            Path object pointing to the generated HTML file
            
        Raises:
            Exception: If notebook export, execution, or HTML conversion fails
        """
        notebook_filepath = Path(notebook_filepath)
        
        # Export the notebook first
        self.export_notebook(str(notebook_filepath), data_path, loading_function)
        
        # Set HTML output path if not provided
        if html_filepath is None:
            html_filepath = notebook_filepath.with_suffix('.html')
        else:
            html_filepath = Path(html_filepath)
        
        # Execute notebook and convert to HTML
        return run_notebook_and_create_html(
            notebook_path=notebook_filepath,
            output_html_path=html_filepath,
            timeout=timeout,
            kernel_name=kernel_name,
            working_directory=notebook_filepath.parent
        )
    
    def run_existing_notebook_to_html(
        self,
        notebook_filepath: str | Path,
        html_filepath: Optional[str | Path] = None,
        timeout: int = 600,
        kernel_name: str = "python3"
    ) -> Path:
        """
        Execute an existing notebook and convert to HTML.
        
        This method can be used to run a previously exported notebook
        and generate an HTML report from it.
        
        Args:
            notebook_filepath: Path to the existing notebook file
            html_filepath: Path for the output HTML file. If None, uses same name as notebook with .html extension
            timeout: Maximum time in seconds to wait for each cell execution (default: 600)
            kernel_name: Name of the Jupyter kernel to use for execution (default: "python3")
            
        Returns:
            Path object pointing to the generated HTML file
            
        Raises:
            FileNotFoundError: If the notebook file doesn't exist
            Exception: If notebook execution or HTML conversion fails
        """
        notebook_filepath = Path(notebook_filepath)
        
        if not notebook_filepath.exists():
            raise FileNotFoundError(f"Notebook file not found: {notebook_filepath}")
        
        # Set HTML output path if not provided
        if html_filepath is None:
            html_filepath = notebook_filepath.with_suffix('.html')
        else:
            html_filepath = Path(html_filepath)
        
        # Execute notebook and convert to HTML
        return run_notebook_and_create_html(
            notebook_path=notebook_filepath,
            output_html_path=html_filepath,
            timeout=timeout,
            kernel_name=kernel_name,
            working_directory=notebook_filepath.parent
        ) 