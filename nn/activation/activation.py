from abc import ABC, abstractmethod, ABCMeta

from nn.propagator.propagator import Propagator


class Activation(Propagator, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()



