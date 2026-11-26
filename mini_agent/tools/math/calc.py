"""Evaluate an arithmetic expression"""
import ast
import operator

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow,
       ast.USub: operator.neg, ast.UAdd: operator.pos}

MAX_BITS = 1_000_000          # about 301,000 decimal digits: anything larger is refused


def _too_large(op, left, right) -> int:
    """The estimated size in bits of an integer power or product that is too large, else 0.

    The estimate comes before the computation, so (10**10000)**10000 is refused at once
    instead of filling memory first.
    """
    if not (type(left) is int and type(right) is int):
        return 0
    if isinstance(op, ast.Pow) and right > 0 and abs(left) > 1:
        bits = (abs(left).bit_length() - 1) * right
    elif isinstance(op, ast.Mult):
        bits = abs(left).bit_length() + abs(right).bit_length()
    else:
        return 0
    return bits if bits > MAX_BITS else 0


def _value(node):
    """Evaluate one node of the parsed expression; anything else is rejected."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_value(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, int) and abs(right) > 10_000:
            raise ValueError("integer exponents above 10000 are not allowed")
        bits = _too_large(node.op, left, right)
        if bits:
            raise ValueError(f"the result would have about {int(bits * 0.30103):,} digits; "
                             f"results above {int(MAX_BITS * 0.30103):,} digits are not computed")
        return OPS[type(node.op)](left, right)
    raise ValueError("use only numbers, + - * / ** ( ) and scientific notation such as 1.6e-19")


def calc(expr: str) -> str:
    """Safely evaluate a math expression"""
    try:
        return str(_value(ast.parse(expr, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"
