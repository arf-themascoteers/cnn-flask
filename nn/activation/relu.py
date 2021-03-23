from nn.activation.activation import Activation


class ReLU(Activation):
    def forward(self, inputs):
        return [max(i, 0) for i in self.inputs]

    def backward(self, dvalues):
        return [int(i > 0) for i in dvalues]


