import numpy as np
n = int(input('Enter number of data points: '))

x = np.zeros((n))
y = np.zeros((n, n))

for i in range(n):
    x[i] = float(input('x=\n'))
    y[i][0] = float(input('y=\n'))
print("backward")

for i in np.arange(1, n):
    for j in range(n - 1, i - 2, -1):
        y[j][i] = y[j][i - 1] - y[j - 1][i - 1]

for i in range(0, n):
    print('%0.2f' % (x[i]), end='')
    for j in range(0, i + 1):
        print('\t%0.2f' % (y[i][j]), end='')
    print()
print("forward:")

for i in range(1, n):
    for j in range(0, n - i):
        y[j][i] = y[j + 1][i - 1] - y[j][i - 1]


for i in range(0, n):
    print('%0.2f' % (x[i]), end='')
    for j in range(0, n - i):
        print('\t\t%0.2f' % (y[i][j]), end='')
    print()