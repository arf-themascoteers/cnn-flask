from abc import ABC, abstractmethod


class Propagator:
    def __init__(self):
        self.inputs = None
        self.dinputs = None
        self.outputs = None

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, dvalues):
        pass

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.outputs = self.forward(inputs)
        return self.outputs

    def backward_pass(self, dvalues):
        self.dinputs = self.backward(dvalues)
        return self.dinputs