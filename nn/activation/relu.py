from nn.activation.activation import Activation

class ReLU(Activation):
    def forward(self, input):
        return [max(i,0) for i in input]

    def backward(self, input):
        return [ int(i>0) for i in input]