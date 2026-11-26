"""The calculator refuses integer results that are too large before computing them."""
import time

from mini_agent.tools.math.calc import calc


def quick(expr):
    t0 = time.monotonic()
    out = calc(expr)
    assert time.monotonic() - t0 < 1.0, f"{expr} took too long"
    return out


def test_nested_powers_are_refused_at_once():
    for expr in ("(10**10000)**10000", "((10**10000)**10000)**10000", "(-(10**10000))**10000"):
        out = quick(expr)
        assert out.startswith("Error: the result would have about"), out
        assert "digits are not computed" in out


def test_a_product_that_is_too_large_is_refused():
    big = "(" + "*".join(["(10**10000)"] * 2) + ")"
    # about 300,000 digits: computed (Python 3.11+ then refuses to print it, 3.10 prints it)
    assert not quick(f"{big}**15").startswith("Error: the result would have")
    out = quick(f"{big}**15 * {big}**15")     # each factor is allowed, the product is not
    assert out.startswith("Error: the result would have about"), out


def test_ordinary_results_are_unchanged():
    assert calc("(2**100)**3") == str(2**300)
    assert calc("2**10") == "1024"
    assert calc("1**100000").startswith("Error: integer exponents")   # the old limit stays first
    assert calc("9**9**9").startswith("Error: integer exponents")
    assert calc("(-1)**10000") == "1"
    assert calc("0**10000") == "0"
    assert abs(float(calc("1.5**100")) - 1.5 ** 100) < 1e-6 * 1.5 ** 100
