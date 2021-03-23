from math import exp

from nn.activation.activation import Activation


class SoftMax(Activation):
    def forward(self, inputs):
        exps = [exp(i) for i in inputs]
        sum_exps = sum(exps)
        exps = [i / sum_exps for i in exps]
        return exps

    def backward(self, dvalues):
        pass
