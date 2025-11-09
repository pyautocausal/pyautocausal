from enum import Enum

class OutputType(Enum):
    """Supported output types for node results"""
    TEXT = "txt"
    CSV = "csv"
    PARQUET = "parquet"
    PNG = "png"
    BINARY = "bin"
    JSON = "json"
    PICKLE = "pkl"

    @classmethod
    def get_supported_extensions(cls):
        return {member.value for member in cls} 