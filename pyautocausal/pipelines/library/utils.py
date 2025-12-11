import pandas as pd

class NodeUtils:
    @staticmethod
    def get_treatment_level(df: pd.DataFrame) -> list[str]:
        """Check at what level treatment is assigned
        take all id_variables and see at what level treatment is assigned e.g. individual county 
        """
        return ["id"]

    def get_fixed_effects(df: pd.DataFrame) -> list[str]:
        """Get fixed effects variables"""
        return ["id", "time"]

    def get_control_variables(df: pd.DataFrame) -> list[str]:
        """Get control variables"""
        return df.columns.drop(["id", "time", "treat"]).tolist()