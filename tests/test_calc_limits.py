"""The calculator's size limits: every integer value (literals and intermediate results included)
is at most MAX_BITS bits, checked exactly at the boundary, and one expression cannot pile up
more than MAX_WORK bits of large integer results in total."""
import ast
import subprocess
import sys
import time

from mini_agent.tools.math.calc import MAX_BITS, _value, calc


def test_a_huge_hex_literal_is_refused():
    expr = "0x1" + "0" * 500_000                  # 2,000,001 bits, not limited by the parser
    try:
        value = _value(ast.parse(expr, mode="eval").body)
    except ValueError:
        return
    assert value.bit_length() <= MAX_BITS, f"accepted a literal of {value.bit_length()} bits"


def test_a_product_of_exactly_the_limit_is_computed():
    term = "((2**10000)**99 * 2**9999)"           # 2**999999: exactly MAX_BITS bits
    assert calc(f"{term}/{term}") == "1.0"


def test_one_bit_over_the_limit_is_refused():
    assert calc("(2**10000)**99 * 2**10000 / 3").startswith("Error: the result would have")


def test_an_intermediate_value_over_the_limit_is_refused_even_when_multiplied_by_zero():
    assert calc("(2**10000)**100 * 0").startswith("Error: the result would have")


def test_many_results_below_the_limit_cannot_add_up_to_unbounded_work():
    terms = ["(9999**10000)**7"] * 256              # each about 930,000 bits
    while len(terms) > 1:
        terms = [f"({terms[i]}+{terms[i + 1]})" for i in range(0, len(terms), 2)]
    code = "import sys; from mini_agent.tools.math.calc import calc; print(calc(sys.stdin.read()))"
    start = time.monotonic()
    done = subprocess.run([sys.executable, "-c", code], input=f"({terms[0]})*0",
                          capture_output=True, text=True, timeout=30)
    elapsed = time.monotonic() - start
    assert done.returncode == 0, done.stderr
    assert "too much computation" in done.stdout, done.stdout[:200]
    assert elapsed < 4, f"took {elapsed:.1f} s"
