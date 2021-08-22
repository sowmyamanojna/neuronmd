import numpy as np
import matplotlib.pyplot as plt
from neuronmd import HHNeuron, IzhNeuron, FHNNeuron

#######################################
# Initialize a HH Neuron Class
hhneuron = HHNeuron()

# tmax - Max time (ms) to simulate the neuron
# dt - time step of simulation
# I - Current for simulating the neuron

tmax = 100
dt = 0.01
I = 0.1
t = np.arange(0, tmax, dt)

hhneuron.simulate(t, I)
hhneuron.plot()

current_list = np.arange(0.01, 0.5, 0.01)
hhneuron.animate(t, ylim=[-85,40], current_list=current_list)
hhneuron.animate(t, current_list=current_list, name="no_ylim")

#######################################
izhneuron = IzhNeuron()

tmax = 1000
dt = 0.01
t = np.arange(0, tmax, dt)
I = 5
izhneuron.simulate(t, I)
izhneuron.plot()

current_list = np.arange(0.01, 10, 0.5)
izhneuron.animate(t, ylim=[-80,35], current_list=current_list)
izhneuron.animate(t, current_list=current_list, name="no_ylim")

#######################################
fhnneuron = FHNNeuron()

tmax = 100
dt = 0.01
I = 1.75
t = np.arange(0, tmax, dt)
fhnneuron.simulate(0.6, 0, t, 0.6)
fhnneuron.plot(name="0.1")

current_list = np.arange(0.01, I, 0.01)
fhnneuron.animate(t, current_list, ylim=[-0.45,1.5])
fhnneuron.animate(t, current_list=current_list, name="no_ylim")
#######################################
