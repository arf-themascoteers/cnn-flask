from abc import ABC, abstractmethod

from nn.activation import Activation


class ReLU(Activation):
    def evaluate(self, value):
        return max(0, value)