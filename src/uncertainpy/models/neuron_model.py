import os

import numpy as np

from model import Model


class NeuronModel(Model):
    def __init__(self,
                 model_file="mosinit.hoc",
                 model_path=None,
                 adaptive_model=False,
                 xlabel="time [ms]",
                 ylabel="voltage [mv]"):

        super(NeuronModel, self).__init__(adaptive_model=adaptive_model,
                                          xlabel=xlabel,
                                          ylabel=ylabel)

        self.model_file = model_file
        self.model_path = model_path



    def load_neuron(self):
        current_dir = os.getcwd()
        os.chdir(self.model_path)

        import neuron

        self.h = neuron.h
        self.h.load_file(1, self.model_file)

        os.chdir(current_dir)



    ### Be really careful with these. Need to make sure that all references to
    ### neuron are inside this class
    def _record(self, ref_data):
        data = self.h.Vector()
        data.record(getattr(self.h, ref_data))
        return data


    def _to_array(self, hocObject):
        array = np.zeros(int(round(hocObject.size())))
        hocObject.to_python(array)
        return array


    def _record_v(self):
        for sec in self.h.allsec():
            self.V = self.h.Vector()
            self.V.record(sec(0.5)._ref_v)
            break


    def _record_t(self):
        self.t = self._record("_ref_t")


    def run(self, **parameters):
        self.load_neuron()

        self.set_parameters(parameters)

        self._record_t()
        self._record_v()

        self.h.run()

        U = self._to_array(self.V)
        t = self._to_array(self.t)

        return t, U



    def set_parameters(self, parameters):
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))
