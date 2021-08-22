import numpy as np
import matplotlib.pyplot as plt
from hh1 import HHNeuron

neuron = HHNeuron()

t = 10
sol = neuron.stimulate(t, 0.2)
print(sol.shape)

plt.figure()
plt.plot(sol[:,0])
plt.title("V")

plt.figure()
plt.plot(sol[:,1])
plt.title("m")

plt.figure()
plt.plot(sol[:,2])
plt.title("n")

plt.figure()
plt.plot(sol[:,3])
plt.title("h")

plt.show()
