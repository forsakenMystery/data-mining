import math


def func(x):
    return x * x * x + x * x - 1


def gunc(x):
    return 1 / math.sqrt(1 + x)


def fixed_point(init, e=10e-8, N=50):
    step = 1
    found = True
    condition = True
    while condition:
        x1 = gunc(init)
        init = x1

        step = step + 1

        if step > N:
            found = False
            break

        condition = abs(func(x1)) > e

    if found:
        print('\nRoot is: %0.2f' % x1)
    else:
        print('\nNot Converged to an answer.')


initial = float(input('Enter your number: \n'))


fixed_point(initial)