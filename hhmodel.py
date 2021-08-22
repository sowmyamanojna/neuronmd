import numpy as np
from math import exp
from scipy.integrate import odeint

class HHNeuron():
    def __init__(self):
        """
        Inititator function that can be 
        used to set the default values
        """
        self.gkmax = 0.36
        self.vk = -77
        self.gnamax = 1.20
        self.vna = 50
        self.gl = 0.003
        self.vl = -54.387
        self.cm = 0.01

        self.v = -64.9964
        self.m = 0.0530
        self.h = 0.5960
        self.n = 0.3177

    def tweak_params(self, params):
        """
        Function to set/change the values of 
        different variables in the model. 
        """
        for key in params:
            setattr(self, key, params[key])

    def integrator(self, z, t):
        """
        The integrator funciton used by odeint
        """
        self.v = z[0]
        self.m = z[1]
        self.n = z[2]
        self.h = z[3]

        self.gna = self.gnamax*self.m**3*self.h
        self.gk = self.gkmax*self.n**4
        self.gtot = self.gna + self.gk + self.gl
        
        na_term = self.gna*self.vna 
        k_term = self.gk*self.vk
        l_term = self.gl*self.vl
        indv_terms = na_term + k_term + l_term
        
        dvdt = (self.current - self.gtot*self.v + indv_terms)/self.cm

        self.alpham = 0.1*(self.v+40)/(1-exp(-(self.v+40)/10))
        self.betam = 4*exp(-0.0556*(self.v+65))

        self.alphan = 0.01*(self.v+55)/(1-exp(-(self.v+55)/10))
        self.betan = 0.125*exp(-(self.v+65)/80)

        self.alphah = 0.07*exp(-0.05*(self.v+65))
        self.betah = 1/(1+exp(-0.1*(self.v+35)))

        dmdt = self.alpham*self.v*(1-self.m) - self.betam*self.v*(self.m)
        dndt = self.alphan*self.v*(1-self.n) - self.betan*self.v*(self.n)
        dhdt = self.alphah*self.v*(1-self.h) - self.betah*self.v*(self.h)
        
        return [dvdt, dmdt, dndt, dhdt]
        


    def stimulate(self, t, current):
        self.dt = t[1] - t[0]
        self.t = t
        self.current = current
        niter = int(self.t[-1]/self.dt)
        self.t_eval = np.arange(0,niter)*self.dt

        z0 = [self.v, self.m, self.n, self.h]
        print(z0)
        # sol = odeint(self.integrator, z0, t, full_output=1)
        sol = odeint(self.integrator, z0, t)
        print(sol)

        return sol