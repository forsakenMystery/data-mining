n = int(input('Enter number of data points: '))

x = []
y = []

print('Enter function points: ')
for i in range(n):
    x.append(float(input('x=\n')))
    y.append(float(input('y=\n')))

xp = float(input('Enter interpolation point: '))

yp = 0
for i in range(n):

    p = 1

    for j in range(n):
        if i != j:
            p = p * (xp - x[j]) / (x[i] - x[j])

    yp += p * y[i]

print('(%.3f, %.3f)' % (xp, yp))