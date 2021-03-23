from abc import ABC, abstractmethod
from math import exp

from nn.activation import Activation

class SoftMax(Activation):
    def evaluate(self,input):
        exps = [exp(i) for i in input]
        sum_exps = sum(exps)
        exps = [i / sum_exps for i in exps]
        return exps