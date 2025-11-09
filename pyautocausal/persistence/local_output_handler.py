from pathlib import Path
from typing import Any
import pandas as pd
import json
import pickle
from .output_handler import OutputHandler
import matplotlib.pyplot as plt
class LocalOutputHandler(OutputHandler):
    """Handles saving outputs to local filesystem"""
    
    def __init__(self, output_dir: str | Path):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_output_path(self, name: str, extension: str) -> Path:
        """Get full output path with extension"""
        output_path = self.output_dir / f"{name}{extension}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def save_csv(self, name: str, data: Any):
        output_path = self._get_output_path(name, ".csv")
        data.to_csv(output_path)
    
    def save_parquet(self, name: str, data: Any):
        output_path = self._get_output_path(name, ".parquet")
        data.to_parquet(output_path)
    
    def save_json(self, name: str, data: Any):
        output_path = self._get_output_path(name, ".json")
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_json(output_path)
        else:
            with open(output_path, 'w') as f:
                json.dump(data, f)
    
    def save_pickle(self, name: str, data: Any):
        output_path = self._get_output_path(name, ".pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    
    def save_text(self, name: str, data: Any):
        output_path = self._get_output_path(name, ".txt")
        with open(output_path, 'w') as f:
            f.write(data)
    
    def save_png(self, name: str, data: Any):
        output_path = self._get_output_path(name, ".png")
        plt.savefig(output_path, format='png')
    
    def save_bytes(self, name: str, data: Any):
        output_path = self._get_output_path(name, ".bytes")
        with open(output_path, 'wb') as f:
            f.write(data) 