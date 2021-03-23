from nn.activation.relu import ReLU
from nn.activation.softmax import SoftMax


class ActivationFactory:
    def createActivation(name):
        if name.lower() == "relu":
            return ReLU()
        if name.lower() == 'softmax':
            return SoftMax()

ActivationFactory.createActivation = staticmethod(ActivationFactory.createActivation)
