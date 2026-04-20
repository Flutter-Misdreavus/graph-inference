"""Restricted code execution sandbox for LLM-generated pandas queries."""
import ast
import time

import pandas as pd


class SecureExecutor:
    """Execute pandas expressions in a restricted environment.

    Only ``df`` (DataFrame) and ``pd`` (pandas module) are exposed.
    Network, filesystem, and arbitrary imports are blocked.
    """

    ALLOWED_NAMES = {
        'df', 'pd', 'len', 'sum', 'max', 'min', 'sorted', 'list', 'set',
        'dict', 'str', 'int', 'float', 'bool', 'range', 'enumerate', 'zip',
        'print', 'isinstance', 'hasattr', 'getattr', 'any', 'all', 'abs',
        'round', 'pow', 'divmod', 'filter', 'map', 'tuple',
    }

    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Call, ast.Name, ast.Constant, ast.Load, ast.Subscript,
        ast.Attribute, ast.Index, ast.Slice, ast.ExtSlice, ast.List,
        ast.Tuple, ast.Dict, ast.Set, ast.IfExp, ast.ListComp,
        ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
        ast.NameConstant, ast.Num, ast.Str, ast.FormattedValue,
        ast.JoinedStr, ast.Starred, ast.keyword, ast.Lambda, ast.arg,
        ast.arguments, ast.Return, ast.Pass, ast.Expr,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.FloorDiv, ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd,
        ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.globals = {'pd': pd, 'df': df}

    def validate(self, code: str) -> ast.Expression:
        """Parse and validate AST; raise SyntaxError if disallowed."""
        try:
            tree = ast.parse(code.strip(), mode='eval')
        except SyntaxError as e:
            raise SyntaxError(f"Invalid syntax: {e}")

        for node in ast.walk(tree):
            if type(node) not in self.ALLOWED_NODES:
                raise SyntaxError(
                    f"Disallowed AST node: {type(node).__name__}"
                )
            if isinstance(node, ast.Name) and node.id not in self.ALLOWED_NAMES:
                raise SyntaxError(
                    f"Disallowed name reference: {node.id}"
                )
            if isinstance(node, ast.Call):
                # Block dangerous builtins via getattr/hasattr
                if isinstance(node.func, ast.Name) and node.func.id in ('eval', 'exec', 'compile', '__import__'):
                    raise SyntaxError(f"Disallowed function: {node.func.id}")

        return tree

    def execute(self, code: str, timeout: float = 5.0):
        """Execute validated code and return result string.

        Returns a tuple (success: bool, result_str: str).
        """
        try:
            tree = self.validate(code)
        except SyntaxError as e:
            return False, f"ValidationError: {e}"

        compiled = compile(tree, '<sandbox>', 'eval')

        start = time.time()
        try:
            result = eval(compiled, self.globals, {})
        except Exception as e:
            return False, f"RuntimeError: {type(e).__name__}: {e}"

        elapsed = time.time() - start
        if elapsed > timeout:
            return False, f"TimeoutError: execution exceeded {timeout}s"

        # Format result for LLM consumption
        if result is None:
            return True, "None"
        if isinstance(result, pd.DataFrame):
            if len(result) > 100:
                return True, f"DataFrame with {len(result)} rows, {len(result.columns)} columns\nFirst 5 rows:\n{result.head().to_string()}"
            return True, result.to_string()
        if isinstance(result, pd.Series):
            if len(result) > 100:
                return True, f"Series with {len(result)} items\nFirst 10:\n{result.head(10).to_string()}"
            return True, result.to_string()
        return True, str(result)
