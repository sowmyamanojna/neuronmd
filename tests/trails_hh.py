import numpy as np
import matplotlib.pyplot as plt
from hhmodel import HHNeuron

neuron = HHNeuron()

tmax = 100
dt = 0.01
I = 0.1
t = np.arange(0, tmax, dt)
# neuron.simulate(t, I)
# neuron.plot(name="0.1")

current_list = np.arange(0.01, 0.5, 0.01)
neuron.animate(t, ylim=[-85,40], current_list=current_list)