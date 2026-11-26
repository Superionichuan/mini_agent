"""Evaluate an arithmetic expression"""
import ast
import operator

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow,
       ast.USub: operator.neg, ast.UAdd: operator.pos}

MAX_BITS = 1_000_000          # about 301,000 decimal digits: any larger integer is refused
MAX_WORK = 8 * MAX_BITS       # the bits of all integer values of one expression together
DIGITS_PER_BIT = 0.30103


def _too_large(bits: int) -> ValueError:
    return ValueError(f"the result would have about {int(bits * DIGITS_PER_BIT):,} digits; "
                      f"results above {int(MAX_BITS * DIGITS_PER_BIT):,} digits are not computed")


def _too_much_work() -> ValueError:
    return ValueError(f"the expression needs too much computation (integer values of more than "
                      f"{int(MAX_WORK * DIGITS_PER_BIT):,} digits in total); compute it in steps")


def _least_bits(op, left, right) -> int:
    """The least number of bits an integer power or product can have, known before it is
    computed, so that (10**10000)**10000 is refused at once instead of filling memory first;
    0 for anything else."""
    if not (type(left) is int and type(right) is int) or left == 0 or right == 0:
        return 0
    if isinstance(op, ast.Pow) and right > 0 and abs(left) > 1:
        return (abs(left).bit_length() - 1) * right + 1
    if isinstance(op, ast.Mult):
        return abs(left).bit_length() + abs(right).bit_length() - 1
    return 0


def _value(node, work=None):
    """Evaluate one node of the parsed expression; anything else is rejected.

    Every integer value, literals and intermediate results included, has at most MAX_BITS
    bits, and all of them together at most MAX_WORK; `work` holds what is left of that for
    the whole expression."""
    if work is None:
        work = [MAX_WORK]
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        value = OPS[type(node.op)](_value(node.operand, work))
    elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = _value(node.left, work), _value(node.right, work)
        if isinstance(node.op, ast.Pow) and isinstance(right, int) and abs(right) > 10_000:
            raise ValueError("integer exponents above 10000 are not allowed")
        least = _least_bits(node.op, left, right)
        if least > MAX_BITS:
            raise _too_large(least)
        if least > work[0]:
            raise _too_much_work()
        value = OPS[type(node.op)](left, right)
    else:
        raise ValueError("use only numbers, + - * / ** ( ) and scientific notation such as 1.6e-19")
    if type(value) is int:
        bits = value.bit_length()
        if bits > MAX_BITS:
            raise _too_large(bits)
        work[0] -= bits
        if work[0] < 0:
            raise _too_much_work()
    return value


def calc(expr: str) -> str:
    """Safely evaluate a math expression"""
    try:
        return str(_value(ast.parse(expr, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"
