import matplotlib.pyplot as plt

x = list(range(1, 562))
y = [xi ** 2 for xi in x]
z = [xi ** 2.5 for xi in x]

plt.plot(x, y, 'r--')
plt.plot(x, z, 'b-')

plt.show()