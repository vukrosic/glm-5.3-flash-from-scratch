"""Fail-closed executable verifier for the synthetic coding tasks."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any

from .tasks import CodingTask

DENIED = (
    ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.With,
    ast.AsyncWith, ast.Lambda, ast.Global, ast.Nonlocal, ast.While, ast.For,
    ast.AsyncFor, ast.Try, ast.Raise, ast.Delete,
)
SAFE_CALLS = {"sum", "len", "abs", "min", "max", "bool", "int", "str"}
SAFE_BUILTINS = {name: __builtins__[name] for name in SAFE_CALLS} if isinstance(__builtins__, dict) else {name: getattr(__builtins__, name) for name in SAFE_CALLS}


@dataclass(frozen=True)
class Evaluation:
    status: str
    passed: bool
    tests_passed: int
    tests_total: int
    pass_fraction: float
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_source(source: str, entry_point: str) -> ast.Module:
    if len(source.encode("utf-8")) > 2048:
        raise ValueError("source too large")
    tree = ast.parse(source)
    functions = []
    for top in tree.body:
        if isinstance(top, ast.Expr) and isinstance(top.value, ast.Constant) and isinstance(top.value.value, str):
            continue
        if isinstance(top, ast.FunctionDef):
            functions.append(top)
            continue
        raise ValueError("top level may contain only the requested function and comments")
    if len(functions) != 1 or functions[0].name != entry_point:
        raise ValueError("required function missing or extra function present")
    for node in ast.walk(tree):
        if isinstance(node, DENIED):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name disallowed")
        if isinstance(node, ast.Attribute):
            raise ValueError("attribute access disallowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_CALLS:
                raise ValueError("unsafe call")
            if node.func.id == entry_point:
                raise ValueError("recursion disallowed")
    return tree


def evaluate_source(task: CodingTask, source: str) -> Evaluation:
    try:
        tree = validate_source(source, task.entry_point)
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        exec(compile(tree, "candidate.py", "exec"), namespace)
        function = namespace[task.entry_point]
    except (SyntaxError, ValueError, TypeError, KeyError) as error:
        return Evaluation("invalid", False, 0, len(task.cases), 0.0, str(error))
    passed = 0
    for arguments, expected in task.cases:
        try:
            actual = function(*arguments)
            passed += int(type(actual) is type(expected) and actual == expected)
        except Exception:
            pass
    total = len(task.cases)
    return Evaluation(
        "passed" if passed == total else "failed",
        passed == total,
        passed,
        total,
        passed / total,
    )


def completion_is_parseable(task: CodingTask, completion: str) -> bool:
    try:
        validate_source(task.prompt + completion, task.entry_point)
        return bool(completion.endswith("\n"))
    except (SyntaxError, ValueError):
        return False
