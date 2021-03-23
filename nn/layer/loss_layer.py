from nn.layer.layer import Layer
import math
import numpy as np


class LossLayer(Layer):
    def __init__(self, output_layer):
        super().__init__(1, None, output_layer, None)
        self.expected_output = None
        self.last_loss = None

    def get_output(self, input):
        logs = list(map(math.log, input))
        products = list(map(math.prod, zip(self.expected_output, logs)))
        self.last_loss = -sum(products)
        return self.last_loss

    def set_expected_output(self, expected_output):
        self.expected_output = expected_output

    def get_derivative(self, back_input):
        return list(map(np.true_divide, zip(self.expected_output, logs)))


