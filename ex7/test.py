import matplotlib.pyplot as plt
import numpy as np

plt.ion()

x = np.arange(0, 4*np.pi, 0.1)
y = [np.sin(i) for i in x]
plt.figure()

for i in range(10):
	plt.plot(x, [j+i for j in y], 'green', linewidth=1.5, markersize=4)
	plt.pause(2.0)
	# input('Press ENTER to continue')

# plt.pause(2.0)
# plt.plot(x, [i**2 for i in y], 'red', linewidth=1.5, markersize=4)
# plt.pause()

plt.ioff()
plt.show()