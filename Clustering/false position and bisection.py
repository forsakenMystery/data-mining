from sympy import var
from sympy import sympify
from sympy import Symbol
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

x = Symbol(input("define your parameter (please use a character 'x'):\n"))
# x = Symbol('x')
# expr = sympify('2*x+4')
# expr = sympify('sin(x)')
expr = sympify(input("define function:\n"))
f = lambdify(x, expr)


def bisection(func, a, b):
    if func(a) * func(b) >= 0:
        return False, a

    c = a
    while (b - a) >= 0.001:

        c = (a + b) / 2

        if func(c) == 0.0:
            break

        if func(c) * func(a) < 0:
            b = c
        else:
            a = c

    return True, c


def falsePosition(func, a, b):
    step = 1
    if func(a) * func(b) >= 0:
        return False, a
    condition = True
    while condition:
        x2 = a - (b - a) * func(a) / (func(b) - func(a))

        if func(a) * func(x2) < 0:
            b = x2
        else:
            a = x2

        step = step + 1
        condition = not func(x2) == 0
        return True, x2


step = 0.1
answers_bisection = []
answers_falsePosition = []
for i in np.arange(-10, 10, step):
    a = i
    b = i + step

    flag, c = bisection(f, a, b)
    if flag:
        answers_bisection.append(float(f"%0.2f" % (c)))

    flag, c = falsePosition(f, a, b)
    if flag:
        answers_falsePosition.append(float(f"%0.2f" % (c)))


print("bisection:", answers_bisection)
print("false position:", answers_falsePosition)


