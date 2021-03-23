from nn.activation.activation import Activation


class ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = [max(i, 0) for i in self.inputs]

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = [int(i > 0) for i in self.dinputs]


