from nn.activation.activation import Activation

class ReLU(Activation):
    def evaluate(self,input):
        return [max(i,0) for i in input]

    def derivative(self, input):
        return [ int(i>0) for i in input]