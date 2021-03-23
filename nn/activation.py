from abc import ABC, abstractmethod

class Activation:
    @abstractmethod
    def evaluate(value):
        pass