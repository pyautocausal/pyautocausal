from pathlib import Path
from typing import Optional, Dict, Any
import nbformat
import subprocess
import sys
import os
import tempfile
import logging

logger = logging.getLogger(__name__)


def run_notebook_and_create_html(
    notebook_path: str | Path,
    output_html_path: Optional[str | Path] = None,
    timeout: int = 600,
    kernel_name: str = "python3",
    working_directory: Optional[str | Path] = None
) -> Path:
    """
    Run a Jupyter notebook and convert it to HTML.
    
    This function executes all cells in a notebook and then converts the
    executed notebook to HTML format with output preserved.
    
    Args:
        notebook_path: Path to the input notebook file (.ipynb)
        output_html_path: Path for the output HTML file. If None, uses same name as notebook with .html extension
        timeout: Maximum time in seconds to wait for each cell execution (default: 600)
        kernel_name: Name of the Jupyter kernel to use for execution (default: "python3")
        working_directory: Directory to run the notebook from. If None, uses notebook's directory
        
    Returns:
        Path object pointing to the generated HTML file
        
    Raises:
        FileNotFoundError: If the notebook file doesn't exist
        subprocess.CalledProcessError: If notebook execution or conversion fails
        ImportError: If required packages (nbconvert, nbclient) are not installed
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook file not found: {notebook_path}")
    
    # Set output HTML path if not provided
    if output_html_path is None:
        output_html_path = notebook_path.with_suffix('.html')
    else:
        output_html_path = Path(output_html_path)
    
    # Set working directory if not provided
    if working_directory is None:
        working_directory = notebook_path.parent
    else:
        working_directory = Path(working_directory)
    
    logger.info(f"Executing notebook: {notebook_path}")
    logger.info(f"Working directory: {working_directory}")
    logger.info(f"Output HTML will be saved to: {output_html_path}")
    
    try:
        # Execute the notebook and convert to HTML in one step
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            "--execute",
            "--ExecutePreprocessor.timeout", str(timeout),
            "--ExecutePreprocessor.kernel_name", kernel_name,
            "--output", str(output_html_path.absolute()),
            str(notebook_path.absolute())
        ]
        
        # Run the command
        result = subprocess.run(
            cmd,
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Successfully executed notebook and created HTML: {output_html_path}")
        return output_html_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to execute notebook or convert to HTML: {e.stderr}"
        logger.error(error_msg)
        raise subprocess.CalledProcessError(e.returncode, e.cmd, e.stdout, error_msg)
    except FileNotFoundError as e:
        if "jupyter" in str(e):
            raise ImportError("Jupyter is not installed. Please install with: pip install jupyter nbconvert nbclient")
        raise


def convert_notebook_to_html(
    notebook_path: str | Path,
    output_html_path: Optional[str | Path] = None,
    include_input: bool = True,
    include_output: bool = True
) -> Path:
    """
    Convert an executed notebook to HTML format.
    
    Args:
        notebook_path: Path to the notebook file (.ipynb)
        output_html_path: Path for the output HTML file. If None, uses same name as notebook with .html extension
        include_input: Whether to include input cells in the HTML (default: True)
        include_output: Whether to include output cells in the HTML (default: True)
        
    Returns:
        Path object pointing to the generated HTML file
        
    Raises:
        FileNotFoundError: If the notebook file doesn't exist
        subprocess.CalledProcessError: If conversion fails
        ImportError: If required packages are not installed
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook file not found: {notebook_path}")
    
    # Set output HTML path if not provided
    if output_html_path is None:
        output_html_path = notebook_path.with_suffix('.html')
    else:
        output_html_path = Path(output_html_path)
    
    logger.info(f"Converting notebook to HTML: {notebook_path}")
    logger.info(f"Output HTML: {output_html_path}")
    
    try:
        # Build the command
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            "--output", str(output_html_path.absolute()),
            str(notebook_path.absolute())
        ]
        
        # Add options for excluding input or output if requested
        if not include_input:
            cmd.extend(["--no-input"])
        if not include_output:
            cmd.extend(["--no-output"])
        
        # Run the command
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Successfully converted notebook to HTML: {output_html_path}")
        return output_html_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to convert notebook to HTML: {e.stderr}"
        logger.error(error_msg)
        raise subprocess.CalledProcessError(e.returncode, e.cmd, e.stdout, error_msg)
    except FileNotFoundError as e:
        if "jupyter" in str(e):
            raise ImportError("Jupyter is not installed. Please install with: pip install jupyter nbconvert")
        raise 