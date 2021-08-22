import numpy as np
import matplotlib.pyplot as plt
from fitzhugh_nagumo import FHNNeuron

neuron = FHNNeuron()

tmax = 100
dt = 0.01
I = 1.75
t = np.arange(0, tmax, dt)
neuron.simulate(0.6, 0, t, 0.6)
neuron.plot(name="0.1")

current_list = np.arange(0.01, I, 0.01)
neuron.animate(t, current_list, ylim=[-0.45,1.5])