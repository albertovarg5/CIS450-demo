import matplotlib
matplotlib.use("TkAgg")   # Force GUI backend

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title("Simple Plot Demo")
plt.xlabel("x")
plt.ylabel("y")

plt.show()
