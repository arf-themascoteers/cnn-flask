from abc import ABC, abstractmethod

class Activation:
    @abstractmethod
    def evaluate(self,input):
        pass

    @abstractmethod
    def derivative(self, input):
        pass