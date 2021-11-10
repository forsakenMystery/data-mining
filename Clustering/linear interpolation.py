x0 = float(input('x0 = '))
y0 = float(input('y0 = '))

x1 = float(input('x1 = '))
y1 = float(input('y1 = '))

xp = float(input('a point to interpolate: '))

print('(%0.4f, %0.4f)' % (xp, y0 + ((y1-y0)/(x1-x0)) * (xp - x0)))