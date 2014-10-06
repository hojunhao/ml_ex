import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C,S = np.cos(X), np.sin(X)

plt.figure(figsize=(20,6), dpi=80)
plt.plot(X,C, 'r--')
plt.plot(X,S)
plt.xticks( [-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks( [-1, 0, 1])
plt.show()