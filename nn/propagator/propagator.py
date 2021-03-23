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


