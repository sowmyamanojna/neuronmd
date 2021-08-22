import numpy as np
import matplotlib.pyplot as plt
from izhikevich import IzhNeuron

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