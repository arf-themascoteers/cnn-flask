from abc import ABC, abstractmethod

class Activation:
    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, input):
        pass