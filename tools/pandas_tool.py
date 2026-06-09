"""Safe Pandas execution for LLM-generated analysis code."""
from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "os", "sys", "subprocess",
    "shutil", "pathlib", "pickle", "builtins", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "help", "memoryview",
}


class UnsafeCodeError(ValueError):
    pass


def _validate_ast(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Imports are not allowed in generated code.")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Forbidden attribute: {node.attr}")


def execute_pandas_code(df: pd.DataFrame, code: str) -> tuple[Any, str]:
    """
    Execute pandas code where the final expression must assign to `result`.
    Example: result = df.groupby('Region')['Sales'].sum()
    """
    code = code.strip()
    if not code:
        raise ValueError("Empty code.")

    if "result" not in code:
        if "\n" not in code and not code.strip().startswith("result"):
            code = f"result = {code}"

    _validate_ast(code)

    namespace: dict[str, Any] = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "result": None,
    }
    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "round": round, "abs": abs, "sorted": sorted, "list": list, "dict": dict,
        "str": str, "int": int, "float": float, "bool": bool, "enumerate": enumerate,
        "zip": zip, "True": True, "False": False, "None": None,
    }
    exec(code, {"__builtins__": safe_builtins}, namespace)
    result = namespace.get("result")
    if result is None:
        raise ValueError("Code must set variable `result` to the answer.")
    return result, code


def result_to_display(result: Any) -> pd.DataFrame | Any:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.reset_index()
    if isinstance(result, (int, float, str, bool, np.number)):
        return pd.DataFrame({"value": [result]})
    try:
        return pd.DataFrame(result)
    except (ValueError, TypeError):
        return pd.DataFrame({"value": [str(result)]})
