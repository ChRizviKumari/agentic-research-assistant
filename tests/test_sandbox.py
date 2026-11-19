import pytest

from research_agent.sandbox import SandboxError, calculator_evaluator, python_evaluator

calc, py = calculator_evaluator(), python_evaluator()


def test_calculator_arithmetic_and_functions():
    assert calc.evaluate("(2+3)*sqrt(16)") == 20
    assert calc.evaluate("2**10 - 1") == 1023
    assert calc.evaluate("max(1, 7, 3) // 2") == 3


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "().__class__.__bases__",
        "open('/etc/passwd')",
        "(lambda: 1)()",
        "[x for x in range(3)]",  # comprehensions disabled in the calculator
        "foo + 1",
        "f'{1}'",
    ],
)
def test_calculator_rejects_forbidden(expr):
    with pytest.raises(SandboxError):
        calc.evaluate(expr)


def test_resource_limits():
    with pytest.raises(SandboxError):
        calc.evaluate("9**9**9")
    with pytest.raises(SandboxError):
        py.evaluate("range(10**9)")
    with pytest.raises(SandboxError):
        py.evaluate("'a' * 10**9")
    with pytest.raises(SandboxError):
        py.evaluate("sum(1 for a in range(90000) for b in range(90000))")


def test_division_by_zero_and_syntax_become_sandbox_errors():
    with pytest.raises(SandboxError):
        calc.evaluate("1/0")
    with pytest.raises(SandboxError):
        calc.evaluate("1 +")


def test_python_evaluator_comprehensions_and_containers():
    assert py.evaluate("sum(x*x for x in range(10) if x % 2 == 0)") == 120
    assert py.evaluate("sorted({'b': 2, 'a': 1})") == ["a", "b"]
    assert py.evaluate("[a+b for a in range(2) for b in range(2)]") == [0, 1, 1, 2]
    assert py.evaluate("[1,2,3][1:]") == [2, 3]
