"""A small AST-walking expression evaluator used by the calculator and python-eval tools.

Instead of ``eval`` with restricted builtins (which is famously escapable), expressions are parsed
with :mod:`ast` and interpreted by an explicit whitelist: only literals, arithmetic, comparisons,
boolean logic, subscripts, comprehensions and calls to *named* whitelisted functions are allowed.
Attribute access, lambdas, f-strings, imports, assignment and dunder names do not exist in this
evaluator. Resource guards cap expression length, evaluation steps, ``range`` size, exponent
size and sequence repetition.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Iterator, Mapping
from typing import Any

MAX_EXPR_CHARS = 2_000
MAX_STEPS = 200_000
MAX_RANGE = 100_000
MAX_POW_BITS = 20_000
MAX_SEQ_REPEAT = 100_000


class SandboxError(Exception):
    """Expression is invalid, forbidden, or exceeded a resource limit."""


def _safe_pow(base: Any, exp: Any) -> Any:
    if isinstance(base, int) and isinstance(exp, int) and exp > 0:
        if max(abs(base).bit_length(), 1) * exp > MAX_POW_BITS:
            raise SandboxError("exponent too large")
    elif isinstance(exp, (int, float)) and abs(exp) > 10_000:
        raise SandboxError("exponent too large")
    return operator.pow(base, exp)


def _safe_mul(a: Any, b: Any) -> Any:
    for seq, n in ((a, b), (b, a)):
        if isinstance(seq, (str, list, tuple)) and isinstance(n, int) and abs(n) > MAX_SEQ_REPEAT:
            raise SandboxError("sequence repetition too large")
    return operator.mul(a, b)


def _safe_range(*args: int) -> range:
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise SandboxError(f"range larger than {MAX_RANGE} elements")
    return r


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: _safe_mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}
_CMPOPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

MATH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
MATH_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

PYTHON_FUNCTIONS: dict[str, Callable[..., Any]] = {
    **MATH_FUNCTIONS,
    "len": len,
    "sum": sum,
    "sorted": sorted,
    "range": _safe_range,
    "any": any,
    "all": all,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "divmod": divmod,
    "pow": _safe_pow,
}


class SafeEvaluator:
    """Interpret a whitelisted subset of Python expressions.

    Args:
        functions: Callable names that expressions may call (by bare name only).
        constants: Extra read-only names.
        allow_comprehensions: Permit list/set/dict comprehensions and generator expressions.
    """

    def __init__(
        self,
        functions: Mapping[str, Callable[..., Any]],
        constants: Mapping[str, Any] | None = None,
        allow_comprehensions: bool = False,
    ) -> None:
        self.functions = dict(functions)
        self.constants = dict(constants or {})
        self.allow_comprehensions = allow_comprehensions
        self._steps = 0

    def evaluate(self, expression: str) -> Any:
        """Evaluate ``expression`` and return the value, or raise :class:`SandboxError`."""
        if len(expression) > MAX_EXPR_CHARS:
            raise SandboxError(f"expression longer than {MAX_EXPR_CHARS} characters")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            raise SandboxError(f"syntax error: {exc.msg}") from exc
        self._steps = 0
        try:
            return self._eval(tree.body, {})
        except SandboxError:
            raise
        except RecursionError as exc:
            raise SandboxError("expression nested too deeply") from exc
        except (ArithmeticError, ValueError, TypeError, IndexError, KeyError) as exc:
            raise SandboxError(f"{type(exc).__name__}: {exc}") from exc

    # -- internals -------------------------------------------------------------------------
    def _tick(self) -> None:
        self._steps += 1
        if self._steps > MAX_STEPS:
            raise SandboxError("evaluation step limit exceeded")

    def _eval(self, node: ast.AST, scope: dict[str, Any]) -> Any:  # noqa: C901 - dispatcher
        self._tick()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str, bool, type(None))):
                return node.value
            raise SandboxError(f"constant of type {type(node.value).__name__} not allowed")
        if isinstance(node, ast.Name):
            if node.id in scope:
                return scope[node.id]
            if node.id in self.constants:
                return self.constants[node.id]
            raise SandboxError(f"name '{node.id}' is not defined")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](self._eval(node.left, scope), self._eval(node.right, scope))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](self._eval(node.operand, scope))
        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And)
            result: Any = None
            for value in node.values:
                result = self._eval(value, scope)
                if bool(result) != is_and:
                    break
            return result
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, scope)
            for op, comparator in zip(node.ops, node.comparators):
                if type(op) not in _CMPOPS:
                    raise SandboxError(f"operator {type(op).__name__} not allowed")
                right = self._eval(comparator, scope)
                if not _CMPOPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            branch = node.body if self._eval(node.test, scope) else node.orelse
            return self._eval(branch, scope)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [self._eval(e, scope) for e in node.elts]
            return items if isinstance(node, ast.List) else tuple(items) if isinstance(
                node, ast.Tuple
            ) else set(items)
        if isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                raise SandboxError("dict unpacking not allowed")
            return {
                self._eval(k, scope): self._eval(v, scope)  # type: ignore[arg-type]
                for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.Subscript):
            return self._eval(node.value, scope)[self._eval(node.slice, scope)]
        if isinstance(node, ast.Slice):
            parts = (node.lower, node.upper, node.step)
            return slice(*[self._eval(p, scope) if p else None for p in parts])
        if isinstance(node, ast.Call):
            return self._call(node, scope)
        if self.allow_comprehensions and isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            return self._comprehension(node, scope)
        raise SandboxError(f"{type(node).__name__} is not allowed")

    def _call(self, node: ast.Call, scope: dict[str, Any]) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in self.functions:
            name = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            raise SandboxError(f"call to '{name}' is not allowed")
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            raise SandboxError("argument unpacking is not allowed")
        args = [self._eval(a, scope) for a in node.args]
        kwargs = {k.arg: self._eval(k.value, scope) for k in node.keywords if k.arg}
        return self.functions[node.func.id](*args, **kwargs)

    def _bind(self, target: ast.AST, value: Any, scope: dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            scope[target.id] = value
        elif isinstance(target, ast.Tuple):
            values = list(value)
            if len(values) != len(target.elts):
                raise SandboxError("cannot unpack comprehension target")
            for sub, item in zip(target.elts, values):
                self._bind(sub, item, scope)
        else:
            raise SandboxError("unsupported comprehension target")

    def _iterate(
        self,
        generators: list[ast.comprehension],
        scope: dict[str, Any],
        depth: int = 0,
    ) -> Iterator[dict[str, Any]]:
        if depth == len(generators):
            yield scope
            return
        gen = generators[depth]
        if gen.is_async:
            raise SandboxError("async comprehension not allowed")
        for item in self._eval(gen.iter, scope):
            self._tick()
            inner = dict(scope)
            self._bind(gen.target, item, inner)
            if all(self._eval(cond, inner) for cond in gen.ifs):
                yield from self._iterate(generators, inner, depth + 1)

    def _comprehension(self, node: Any, scope: dict[str, Any]) -> Any:
        scopes = self._iterate(node.generators, dict(scope))
        if isinstance(node, ast.DictComp):
            return {self._eval(node.key, s): self._eval(node.value, s) for s in scopes}
        values = [self._eval(node.elt, s) for s in scopes]
        return set(values) if isinstance(node, ast.SetComp) else values


def calculator_evaluator() -> SafeEvaluator:
    """Evaluator limited to numeric math (no comprehensions or containers needed)."""
    return SafeEvaluator(MATH_FUNCTIONS, MATH_CONSTANTS, allow_comprehensions=False)


def python_evaluator() -> SafeEvaluator:
    """Evaluator for general (still sandboxed) Python expressions."""
    return SafeEvaluator(PYTHON_FUNCTIONS, MATH_CONSTANTS, allow_comprehensions=True)
