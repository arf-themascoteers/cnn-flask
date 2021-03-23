from abc import ABC, abstractmethod

from nn.activation import Activation

class ReLU(Activation):
    def evaluate(self,input):
        return [max(i,0) for i in input]