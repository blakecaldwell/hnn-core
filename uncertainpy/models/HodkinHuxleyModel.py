from model import Model

import numpy as np
import odespy

# The class name and file name must be the same
class HodkinHuxleyModel(Model):
    """
    The model must be able to handle these calls

    simulation = model()
    simulation.load()
    simulation.setParameters(parameters -> dictionary)
    simulation.run()
    simulation.save(current_process -> int)

    simulation.cmd()
    """
    def __init__(self, parameters=None):
        """
        Init must be able to be called with 0 arguments
        """
        Model.__init__(self, parameters=parameters)


        ## setup parameters and state variables
        self.T = 45    # ms
        self.dt = 0.025  # ms
        self.t = np.arange(0, self.T + self.dt, self.dt)

        ## HH Parameters
        self.V_rest = -65   # mV
        self.Cm = 1         # uF/cm**2
        self.gbar_Na = 120  # mS/cm**2
        self.gbar_K = 36    # mS/cm**2
        self.gbar_l = 0.3   # mS/cm**2
        self.E_Na = 50      # mV
        self.E_K = -77      # mV
        self.E_l = -54.4    # mV
        self.m0 = 0.0011    # unitless
        self.n0 = 0.0003    # unitless
        self.h0 = 0.9998    # unitless


        self.xlabel = "time [ms]"
        self.ylabel = "voltage [mv]"

        self.I = lambda t: 0

    # def I(self, t):
    #     if t >= 5 and t <= 30:
    #         return 10
    #     else:
    #         return 0

    # K channel
    def alpha_n(self, V):
        return 0.01*(V + 55)/(1 - np.exp(-(V + 55)/10.))

    def beta_n(self, V):
        return 0.125*np.exp(-(V + 65)/80.)

    def n_f(self, n, V):
        return self.alpha_n(V)*(1 - n) - self.beta_n(V)*n


    # Na channel (activating)
    def alpha_m(self, V):
        return 0.1*(V + 40)/(1 - np.exp(-(V + 40)/10.))

    def beta_m(self, V):
        return 4*np.exp(-(V + 65)/18.)

    def m_f(self, m, V):
        return self.alpha_m(V)*(1 - m) - self.beta_m(V)*m


    # Na channel (inactivating)
    def alpha_h(self, V):
        return 0.07*np.exp(-(V + 65)/20.)

    def beta_h(self, V):
        return 1/(np.exp(-(V + 35)/10.) + 1)

    def h_f(self, h, V):
        return self.alpha_h(V)*(1 - h) - self.beta_h(V)*h


    def dXdt(self, X, t):
        V, h, m, n = X

        g_Na = self.gbar_Na*(m**3)*h
        g_K = self.gbar_K*(n**4)
        g_l = self.gbar_l

        dmdt = self.m_f(m, V)
        dhdt = self.h_f(h, V)
        dndt = self.n_f(n, V)

        print t
        print self.I(t)

        dVdt = (self.I(t) - g_Na*(V - self.E_Na) - g_K*(V - self.E_K) - g_l*(V - self.E_l))/self.Cm

        return [dVdt, dhdt, dmdt, dndt]


    def run(self):

        solver = odespy.RK4(self.dXdt)
        solver.set_initial_condition([self.V_rest, self.h0, self.m0, self.n0])

        X, t = solver.solve(self.t)

        self.t = t
        self.U = X[:, 0]


    # def run(self):
    #
    #     Vm = np.zeros(len(self.t))  # mV
    #     Vm[0] = self.V_rest
    #
    #     m = self.m0
    #     h = self.h0
    #     n = self.n0
    #
    #
    #     for i in range(1, len(self.t)):
    #         g_Na = self.gbar_Na*(m**3)*h
    #         g_K = self.gbar_K*(n**4)
    #         g_l = self.gbar_l
    #
    #         m += self.dt*self.m_f(m, Vm[i-1])
    #         h += self.dt*self.h_f(h, Vm[i-1])
    #         n += self.dt*self.n_f(n, Vm[i-1])
    #
    #         Vm[i] = Vm[i-1] + (self.I[i-1] - g_Na*(Vm[i-1] - self.E_Na) \
    #                            - g_K*(Vm[i-1] - self.E_K) - g_l*(Vm[i-1] - self.E_l))/self.Cm*self.dt
    #
    #
    #     self.U = Vm
