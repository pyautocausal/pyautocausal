"""Serialize the example_graph pipeline to YAML for deployment.

Run this script from the project root or the *scripts/* directory:

    python scripts/serialize_pipeline.py

It produces ``scripts/example_graph.yml`` which can be uploaded to S3 and
consumed by the FastAPI service.
"""

from pathlib import Path
import sys

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from the 'scripts' directory
# or the project root.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pyautocausal.pipelines.example_graph import simple_graph

def serialize_graph_to_file():
    """Build the example graph and write it to *example_graph.yml*."""
    # We need to provide a dummy output path for the graph instantiation,
    # though it's not used when we just serialize the graph structure.
    dummy_output_path = Path("./temp_output")
    dummy_output_path.mkdir(exist_ok=True)

    print("Instantiating the example graph pipeline...")
    pipeline_graph = simple_graph(output_path=dummy_output_path)

    # Define the output file path in the 'scripts' directory
    output_filename = "example_graph.yml"
    output_filepath = Path(__file__).resolve().parent / output_filename
    
    print(f"Serializing pipeline object to: {output_filepath}")
    
    try:
        # Serialize to YAML
        pipeline_graph.to_yaml(output_filepath)
        
        print(f"Successfully serialized pipeline to '{output_filepath}'.")
        print("You can now manually upload this file to your S3 bucket.")

    except Exception as e:
        print(f"Error during serialization: {e}")
        print("The pipeline object could not be serialized.")
    finally:
        # Clean up the dummy directory
        try:
            # Clean up the dummy directory and any files inside it
            for item in dummy_output_path.iterdir():
                item.unlink()
            dummy_output_path.rmdir()
        except OSError as e:
            print(f"Note: Could not remove temporary directory '{dummy_output_path}': {e}")


if __name__ == "__main__":
    serialize_graph_to_file() 